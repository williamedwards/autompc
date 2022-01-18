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

from .optimizer import Optimizer


class MultivariateNormal:
    def __init__(self, mu, cov):
        self.scale = np.sqrt(cov)
        # np.random.seed(42)

    def sample(self, shape):
        new_shape = shape + (1,)
        noise = np.random.normal(scale=self.scale, size=new_shape)
        return noise

class MPPI(Optimizer):
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
    # FIXME Random seeds not properly controlled
    def __init__(self, system):
        super().__init__(system, "MPPI")

    def get_default_config_space(self):
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

    def set_config(self, config):
        self.H = config["horizon"]
        self.sigma = config["sigma"]  # sigma of the normal distribution
        self.lmda = config["lmda"]
        self.num_path = config["num_path"]

    def set_ocp(self, ocp):
        super().set_ocp(ocp)
        cost = ocp.get_cost()
        def cost_eqn(path, actions):
            costs = np.zeros(path.shape[0])
            for i in range(path.shape[0]):
                costs[i] += cost.eval_obs_cost(path[i,:self.system.obs_dim])
                costs[i] += cost.eval_ctrl_cost(actions[i,:])
            return costs
        def terminal_cost(path):
            last_obs = path[-1, :self.system.obs_dim]
            term_cost = cost.eval_term_obs_cost(last_obs)
            return term_cost
        self.cost_eqn = cost_eqn
        self.terminal_cost = terminal_cost

    def set_model(self, model):
        super().set_model(model)
        self.dyn_eqn = model.pred_batch

    def reset(self):
        self.dim_state, self.dim_ctrl = self.model.state_dim, self.system.ctrl_dim
        self.act_sequence = np.zeros((self.H, self.dim_ctrl))  
        self.noise_dist = MultivariateNormal(0, self.sigma)
        self.act_sequence = self.noise_dist.sample((self.H,))
        self.umin = self.ocp.get_ctrl_bounds()[:,0]
        self.umax = self.ocp.get_ctrl_bounds()[:,1]
        self.ctrl_scale = self.umax
        self.niter = 1 # FIXME ?

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

    def run(self, state):
        x0 = state
        # then collect trajectories...
        for _ in range(self.niter):
            costs, eps = self.do_rollouts(x0)
            self.update(costs, eps)
        # update the cached action sequence
        ret_action = self.act_sequence[0].copy()
        ret_action *= self.ctrl_scale

        return ret_action

    def get_state(self):
        # TODO add random seed to state
        return {"act_sequence", copy.deepcopy(act_sequence)}

    def set_state(self, state):
        self.act_sequence = copy.deepcopy(state["act_sequence"])

    def is_compatible(self, system, task, model):
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
