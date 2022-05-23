
# Standard library includes
import copy
import multiprocessing as mp

# External library includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

# Internal libary includes
from .optimizer import Optimizer


class MultivariateNormal:
    def __init__(self, mu, cov):
        self.scale = np.sqrt(cov)

    def sample(self, shape, rng):
        new_shape = shape + (1,)
        noise = rng.normal(scale=self.scale, size=new_shape)
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

    Parameters:

    - **niter** *(Type: int, Default: 1)*: Number of MPPI iterations to perform at each time step.

    Hyperparameters:

    - **horizon** *(Type: int, Lower: 5, Upper: 30, Default: 20)*: Controller horizon. This behaves in the same way as MPC controller.
    - **sigma** *(Type: float, Lower: 0.1, Upper 2.0, Default: 1.0)*: Variance of the disturbance. Since the disturbance is Gaussian, the standard deviation is its square root. It is shared in all the control dimensions.
    - **lmda** *(Type: float, Lower: 10^-4, Upper: 2.0, Default: 1.0)*: Higher value increases the cost of control noise and gets more samples around current contorl sequence. 
        Generally smaller value works better.
    - **num_path** *(Type: int, Lower: 100, Upper: 1000, Default: 200)*: Number of perturbed control sequence to sample. Generally the more the better and it scales better with vectorized and parallel computation.
    """
    # FIXME Random seeds not properly controlled
    def __init__(self, system, niter=1):
        super().__init__(system, "MPPI")
        self.niter = niter

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
        self.horizon = config["horizon"]
        self.sigma = config["sigma"]  # sigma of the normal distribution
        self.lmda = config["lmda"]
        self.num_path = config["num_path"]

    def set_ocp(self, ocp):
        super().set_ocp(ocp)
        cost = ocp.get_cost()
        def cost_eqn(path, actions):
            costs = np.array([cost.incremental(path[i,:self.system.obs_dim],actions[i,:])*self.system.dt for i in range(path.shape[0])])
            return costs
        def terminal_cost(path):
            term_costs = np.array([cost.terminal(path[i, :self.system.obs_dim]) for i in range(path.shape[0])])
            return term_costs
        self.cost_eqn = cost_eqn
        self.terminal_cost = terminal_cost

    def set_model(self, model):
        super().set_model(model)
        self.dyn_eqn = model.pred_batch

    def reset(self, seed=100):
        H = self.horizon
        self.rng = np.random.default_rng(seed)
        self.dim_state, self.dim_ctrl = self.model.state_dim, self.system.ctrl_dim
        self.act_sequence = np.zeros((H, self.dim_ctrl))  
        self.noise_dist = MultivariateNormal(0, self.sigma)
        self.act_sequence = self.noise_dist.sample((H,), self.rng)
        self.umin = self.ocp.get_ctrl_bounds()[:,0]
        self.umax = self.ocp.get_ctrl_bounds()[:,1]
        self.ctrl_scale = self.umax

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
        H = self.horizon
        # generate random noises
        # eps = np.random.normal(scale=self.sigma, size=(H, self.num_path, self.dim_ctrl))  # horizon by num_path by ctrl_dim
        eps = self.noise_dist.sample((self.num_path, H), self.rng).transpose((1, 0, 2))
        # path = np.zeros((H + 1, self.num_path, self.dim_state))  # horizon by num_path by state_dim
        # path[0] = cur_state  # just copy the initial state in...
        path = np.zeros((self.num_path, self.dim_state))
        path[:] = cur_state
        costs = np.zeros(self.num_path)
        action_cost = np.zeros_like(costs)
        for i in range(H):
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

    def step(self, state):
        x0 = state
        # advance the action sequence from last iteration
        self.act_sequence[:-1] = self.act_sequence[1:]
        self.act_sequence[-1] = self.act_sequence[-2]
        # then collect trajectories...
        for _ in range(self.niter):
            costs, eps = self.do_rollouts(x0)
            self.update(costs, eps)
        # update the cached action sequence
        ret_action = self.act_sequence[0].copy()
        ret_action *= self.ctrl_scale

        return ret_action

    def get_state(self):
        return {"act_sequence" : copy.deepcopy(self.act_sequence),
                "rng_state" : self.rng.bit_generator.__getstate__()}

    def set_state(self, state):
        self.act_sequence = copy.deepcopy(state["act_sequence"])
        self.rng.bit_generator.__setstate__(state["rng_state"])

    def is_compatible(self, model, ocp):
        return True