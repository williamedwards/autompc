"""
Implement the model predictive path integral control methods.
The implementation is based on paper Information theoretic MPC for model-based reinforcement learning
from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7989202
It directly modifies code from github repository called pytorch_mppi but now uses numpy
"""
import numpy as np
import multiprocessing as mp
import copy
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from .controller import Controller, ControllerFactory


class MultivariateNormal:
    def __init__(self, mu, cov):
        self.scale = np.sqrt(cov)
        # np.random.seed(42)

    def sample(self, shape):
        new_shape = shape + (1,)
        noise = np.random.normal(scale=self.scale, size=new_shape)
        return noise

class MPPIFactory(ControllerFactory):
    """
    Implementation of Model Predictive Path Integral (MPPI) controller.
    It originates from stochastic optimal control but also works for deterministic systems.
    Starting from some initial guess of control sequence, it samples some perturbed control sequences,
    and use the costs from each rollout of the perturbed control to update the control sequence.
    For details, we refer to `Aggressive driving with model predictive path integral control <https://ieeexplore.ieee.org/document/7487277/>`_,
    `Model Predictive Path Integral Control: From Theory to Parallel Computation <https://arc.aiaa.org/doi/abs/10.2514/1.G001921>`_,
    and `Information Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving <https://arxiv.org/abs/1707.02342>`_.
    Development of this controller uses reference from `this repository <https://github.com/UM-ARM-Lab/pytorch_mppi>`_.

    Hyperparameters:

    - *horizon* (Type: int, Lower: 5, Upper: 30, Default: 20): Controller horizon. This behaves in the same way as MPC controller.
    - *sigma* (Type: float, Lower: 0.1, Upper 2.0, Default: 1.0): Variance of the disturbance. Since the disturbance is Gaussian, the standard deviation is its square root. It is shared in all the control dimensions.
    - *lmda* (Type: float, Lower: 10^-4, Upper: 2.0, Default: 1.0): Higher value increases the cost of control noise and gets more samples around current contorl sequence. 
        Generally smaller value works better.
    - *num_path* (Type: int, Lower: 100, Upper: 1000, Default: 200): Number of perturbed control sequence to sample. Generally the more the better and it scales better with vectorized and parallel computation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.Controller = MPPI
        self.name = "MPPI"

    def get_configuration_space(self):
        cs = CS.ConfigurationSpace()
        horizon = CSH.UniformIntegerHyperparameter(name="horizon",
                lower=5, upper=30, default_value=20)
        cs.add_hyperparameter(horizon)
        sigma = CSH.UniformFloatHyperparameter(name='sigma', lower=1e-4, upper=2.0, 
                default_value=1.0)
        cs.add_hyperparameter(sigma)
        lmda = CSH.UniformFloatHyperparameter(name='lmda', lower=0.1, upper=2.0, 
                default_value=1.0)
        cs.add_hyperparameter(lmda)
        num_path = CSH.UniformIntegerHyperparameter(name='num_path', lower=100, 
                upper=1000, default_value=200)
        cs.add_hyperparameter(num_path)
        return cs

class MPPI:
    def __init__(self, system, task, model, **kwargs):
        self.kwargs = kwargs 
        self.model = model
        self.dyn_eqn = model.pred_parallel
        cost = task.get_cost()
        def cost_eqn(path, actions):
            costs = np.zeros(path.shape[0])
            for i in range(path.shape[0]):
                costs[i] += cost.eval_obs_cost(path[i,:])
                costs[i] += cost.eval_ctrl_cost(actions[i,:])
            return costs
        def terminal_cost(path):
            return cost.eval_term_obs_cost(path[-1])
        self.cost_eqn = cost_eqn
        self.terminal_cost = terminal_cost
        system = model.system
        self.dim_state, self.dim_ctrl = system.obs_dim, system.ctrl_dim
        self.seed = kwargs.get('seed', 0)
        self.H = kwargs.get('horizon', 20)
        print(f"H={self.H}")
        self.num_path = kwargs.get('num_path', 1000)  # how many paths are generated in parallel
        print(f"num_path={self.num_path}")
        self.num_iter = kwargs.get('niter', 1)
        self.sigma = kwargs.get('sigma', 1)  # sigma of the normal distribution
        self.lmda = kwargs.get('lmda', 1.0)  # scale the cost...
        print(f"sigma={self.sigma}")
        print(f"lmda={self.lmda}")
        self.act_sequence = np.zeros((self.H, self.dim_ctrl))  # set initial default action as zero
        self.noise_dist = MultivariateNormal(0, self.sigma)
        self.act_sequence = self.noise_dist.sample((self.H,))
        self.umin = task.get_ctrl_bounds()[:,0]
        self.umax = task.get_ctrl_bounds()[:,1]
        self.ctrl_scale = self.umax
        # for the seed
        self.cur_step = 0
        self.niter = 1

    def reset(self):
        self.__init__(self, self.system, self.task, self.model, **self.kwargs)

    def update(self, costs, eps):
        """Based on the collected trajectory, update the action sequence.
        costs is of shape num_path
        eps is of shape H by num_path by dimu
        """
        S = np.exp(-1 / self.lmda * (costs - np.amin(costs)))
        weight = S / np.sum(S)
        update = np.sum(eps * weight[None, :, None], axis=1)  # so update of shape H by dimu
        self.act_sequence += update

    def do_rollouts(self, cur_state, seed=None):
        # roll the action
        self.act_sequence[:-1] = self.act_sequence[1:]
        self.act_sequence[-1] = self.act_sequence[-2]
        # generate random noises
        # eps = np.random.normal(scale=self.sigma, size=(self.H, self.num_path, self.dim_ctrl))  # horizon by num_path by ctrl_dim
        eps = self.noise_dist.sample((self.num_path, self.H)).transpose((1, 0, 2))
        # path = np.zeros((self.H + 1, self.num_path, self.dim_state))  # horizon by num_path by state_dim
        # path[0] = cur_state  # just copy the initial state in...
        path = np.zeros((self.num_path, self.dim_state))
        path[:] = cur_state
        costs = np.zeros(self.num_path)
        action_cost = np.zeros_like(costs)
        for i in range(self.H):
            actions = eps[i] + self.act_sequence[i]
            # bound actions if necessary
            if self.umin is not None and self.umax is not None:
                actions = np.minimum(self.umax/self.ctrl_scale, 
                        np.maximum(self.umin/self.ctrl_scale, actions))
                eps[i] = actions - self.act_sequence[i]
            # costs += self.cost_eqn(path[i], actions) + self.lmda / self.sigma * np.einsum('ij,ij->i', actions, eps[i])
            # path[i + 1] = self.dyn_eqn(path[i], actions)
            costs += self.cost_eqn(path, actions*self.ctrl_scale)
            action_cost += self.lmda / self.sigma * np.einsum('ij,ij->i', actions, eps[i])
            path = self.dyn_eqn(path, actions*self.ctrl_scale)
        # the final cost
        if self.terminal_cost:
            # costs += self.terminal_cost(path[-1])
            costs += self.terminal_cost(path)
        # print('state = ', path, 'cost path = ', costs, 'pert_cost = ', action_cost)
        costs += action_cost
        # import pdb; pdb.set_trace()
        return costs, eps

    @property
    def state_dim(self):
        return 0

    def traj_to_state(self, traj):
        return np.array([])

    def run(self, constate, new_obs):
        # first is to extract current state
        x0 = new_obs
        # then collect trajectories...
        for _ in range(self.niter):
            costs, eps = self.do_rollouts(x0, self.seed + self.cur_step)
            self.update(costs, eps)
        self.cur_step += 1
        # update the cached action sequence
        ret_action = self.act_sequence[0].copy()
        ret_action *= self.ctrl_scale
        return ret_action, np.array([])

    def traj_to_state(self, traj):
        return self.model.traj_to_state(traj)

    @property
    def state_dim(self):
        return self.model.state_dim

    @staticmethod
    def is_compatible(system, task, model):
        # MPPI works with all model/system/task unless there is hard constraints, but can't we?
        return True


class MPPICopy(Controller):
    def __init__(self, system, task, model, **kwargs):
        Controller.__init__(self, system, task, model)
        self.n, self.m = system.obs_dim, system.ctrl_dim
        self.seed = kwargs.get('seed', 0)
        self.H = kwargs.get('H', 20)
        self.num_path = kwargs.get('num_path', 1000)
        self.num_iter = kwargs.get('niter', 1)
        self.num_cpu = kwargs.get('num_cpu', 10)
        betas = kwargs.get('filter_coefs', {'beta_0': 0.25, 'beta_1': 0.8, 'beta_2': 0.0})
        self.sigma = kwargs.get('sigma', 1)
        self.filter_coefs = (self.sigma, betas['beta_0'], betas['beta_1'], betas['beta_2'])
        self.lmda = kwargs.get('lmda', 1.0)  # scale the cost...
        self.paths_per_cpu = int(np.ceil(self.num_path / self.num_cpu))

        self.act_sequence = np.zeros((self.H, self.m))  # set initial default action as zero

        # for the seed
        self.cur_step = 0
        self.niter = 1
    
    def update(self, paths):
        """Based on the collected trajectory, update the action sequence"""
        num_traj = len(paths)
        act = np.array([paths[i]["actions"] for i in range(num_traj)])
        R = self.score_trajectory(paths)
        # jlmd = (np.amax(R) - np.amin(R)) / 10
        # jif lmd < 1:
        # j    lmd = 1
        lmd = self.lmda
        S = np.exp(-1 / lmd * (R - np.amin(R)))
        # blend the action sequence
        weighted_seq = S * act.T
        act_sequence = np.sum(weighted_seq.T, axis=0)/(np.sum(S) + 1e-6)
        self.act_sequence = act_sequence

    def advance_time(self, act_sequence=None):
        """In this function, the cached act_sequence is updated"""
        self.act_sequence[:-1] = self.act_sequence[1:]
        self.act_sequence[-1] = self.act_sequence[-2]  # just copy the control...

    def score_trajectory(self, paths):
        """Score trajectories, here I use negative cost as reward"""
        scores = np.array([path['cost'] for path in paths])
        return scores

    def do_rollouts(self, cur_state, seed=None):
        paths = gather_paths_parallel(self.model,
                                      self.task,
                                      cur_state,
                                      self.act_sequence,
                                      self.filter_coefs,
                                      seed,
                                      self.paths_per_cpu,
                                      self.num_cpu,
                                      )
        return paths

    def run(self, traj, latent=None):
        # first is to extract current state
        x0 = self.model.traj_to_state(traj)
        # then collect trajectories...
        for _ in range(self.niter):
            paths = self.do_rollouts(x0, self.seed + self.cur_step)
            self.update(paths)
        self.cur_step += 1
        # update the cached action sequence
        ret_action = self.act_sequence[0].copy()
        self.advance_time()
        return ret_action, None

    def traj_to_state(self, traj):
        return self.model.traj_to_state(traj)

    @property
    def state_dim(self):
        return self.model.state_dim

    @staticmethod
    def get_configuration_space(system, task, model):
        cs = CS.ConfigurationSpace()
        horizon = CSH.UniformIntegerHyperparameter(name="horizon",
                lower=10, upper=100, default_value=10)
        cs.add_hyperparameter(horizon)
        kappa = CSH.UniformFloatHyperparameter(name='kappa', lower=0.1, upper=1.0, default_value=1.0)
        cs.add_hyperparameter(kappa)
        num_traj = CSH.UniformIntegerHyperparameter(name='num_traj', lower=100, upper=1000, default_value=200)
        cs.add_hyperparameter(num_traj)
        return cs

    @staticmethod
    def is_compatible(system, task, model):
        # MPPI works with all model/system/task unless there is hard constraints, but can't we?
        return True
 

def do_env_rollout(model, task, start_state, act_list):
    """
        1) Construct env with env_id and set it to start_state.
        2) Generate rollouts using act_list.
           act_list is a list with each element having size (H,m).
           Length of act_list is the number of desired rollouts.
    """
    e = copy.copy(model)
    paths = []
    H = act_list[0].shape[0]
    N = len(act_list)
    add_obs_cost, add_ctrl_cost, terminal_obs_cost = task.get_costs()
    for i in range(N):  # repeat simulation for this long...
        state = start_state.copy()
        act = []
        states = []
        reward = 0
        for k in range(H):
            ctrl = act_list[i][k]
            act.append(ctrl)
            states.append(state)
            next_state = e.pred(state, ctrl)
            reward += (add_obs_cost(state) + add_ctrl_cost(ctrl)) * model.dt
            state = next_state
        reward += terminal_obs_cost(state)  # terminal cost evaluated at the final state, nice...

        path = dict(
                    actions=np.array(act),
                    cost=reward,
                    states=np.array(states),
                    statef=state
                    )
        paths.append(path)

    return paths


def generate_perturbed_actions(base_act, filter_coefs):
    """
    Generate perturbed actions around a base action sequence
    """
    sigma, beta_0, beta_1, beta_2 = filter_coefs
    eps = np.random.normal(loc=0, scale=1.0, size=base_act.shape) * sigma
    for i in range(2, eps.shape[0]):
        eps[i] = beta_0*eps[i] + beta_1*eps[i-1] + beta_2*eps[i-2]
    return base_act + eps


def generate_paths(model, task, start_state, N, base_act, filter_coefs, base_seed):
    """
    first generate enough perturbed actions
    then do rollouts with generated actions
    set seed inside this function for multiprocessing
    """
    np.random.seed(base_seed)
    act_list = []
    for i in range(N):
        act = generate_perturbed_actions(base_act, filter_coefs)
        act_list.append(act)
    paths = do_env_rollout(model, task, start_state, act_list)
    return paths


def generate_paths_star(args_list):
    return generate_paths(*args_list)

# here env_id should be the model
def gather_paths_parallel(model, task, start_state, base_act, filter_coefs, base_seed, paths_per_cpu, num_cpu=None):
    num_cpu = mp.cpu_count() if num_cpu is None else num_cpu
    args_list = []
    for i in range(num_cpu):
        cpu_seed = base_seed + i*paths_per_cpu
        args_list_cpu = [model, task, start_state, paths_per_cpu, base_act, filter_coefs, cpu_seed]
        args_list.append(args_list_cpu)

    # do multiprocessing
    results = _try_multiprocess(args_list, num_cpu, max_process_time=300, max_timeouts=4)
    paths = []
    for result in results:
        for path in result:
            paths.append(path)

    return paths


def _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts):
    # Base case
    if max_timeouts == 0:
        return None

    if num_cpu == 1:
        results = [generate_paths_star(args_list[0])]  # dont invoke multiprocessing unnecessarily

    else:
        pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
        parallel_runs = [pool.apply_async(generate_paths_star,
                                         args=(args_list[i],)) for i in range(num_cpu)]
        try:
            results = [p.get(timeout=max_process_time) for p in parallel_runs]
        except Exception as e:
            print(str(e))
            print("Timeout Error raised... Trying again")
            pool.close()
            pool.terminate()
            pool.join()
            return _try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts - 1)

        pool.close()
        pool.terminate()
        pool.join()

    return results
