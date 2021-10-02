import numpy as np
from .surrogate_evaluator import SurrogateEvaluator
from .control_evaluator import ControlEvaluator, NormalDistribution
from ..utils import simulate
from ..evaluation.model_metrics import get_model_residuals
from ..sysid.model import Model
from ..control.controller import Controller

class StochasticSurrogate(Model):
    def __init__(self, model, noise_mean, noise_cov, rng):
        super().__init__(model.system)
        self.model = model
        self.rng = rng
        self.noise_mean = noise_mean
        self.noise_cov = noise_cov

    def traj_to_state(self, traj):
        return self.model.traj_to_state(traj)
    
    def update_state(self, *args, **kwargs):
        return self.model.update_state(*args, **kwargs)

    @property
    def state_dim(self):
        return self.model.state_dim

    def pred(self, state, ctrl):
        xpred = self.pred_batch(state.reshape((1,state.size)), 
                ctrl.reshape((1,ctrl.size)))[0,:]
        return xpred

    def pred_batch(self, states, ctrls):
        xpred = self.model.pred_batch(states, ctrls)
        noise = self.rng.multivariate_normal(self.noise_mean, self.noise_cov, xpred.shape[0])
        return xpred + noise

class StochasticSurrogateEvaluator(SurrogateEvaluator):
    def __init__(self, system, task, trajs, surrogate_factory, rng, n_sims=10, surrogate_cfg=None,
                    surrogate_mode="defaultcfg", surrogate_tune_iters=100, holdout_prop=0.1):
        ControlEvaluator.__init__(self, system, task, trajs)
        self.surrogate_mode = surrogate_mode
        self.surrogate_factory = surrogate_factory
        self.seeds = [rng.integers(1 << 31) for _ in range(n_sims)]

        # Holdout evaluation data
        rng = np.random.default_rng(100)
        holdout_size = int(holdout_prop * len(trajs))
        shuffled_trajs = trajs[:]
        rng.shuffle(shuffled_trajs)
        self.holdout_trajs = shuffled_trajs[:holdout_size]
        self.surr_trajs = shuffled_trajs[holdout_size:]

        self.surrogate, self.surr_tune_result = self._get_surrogate(self.surr_trajs, rng, surrogate_tune_iters)

        residuals = get_model_residuals(self.surrogate, self.holdout_trajs, horizon=1)
        self.resid_mean = np.mean(residuals, axis=0)
        self.resid_cov  = np.cov(residuals.T)

    def _run_sim(self, controller, seed):
        rng = np.random.default_rng(seed)
        surrogate = StochasticSurrogate(self.surrogate, self.resid_mean, self.resid_cov, rng)
        try:
            controller.reset()
            if self.task.has_num_steps():
                surr_traj = simulate(controller, self.task.get_init_obs(),
                    self.task.term_cond, sim_model=surrogate, 
                    ctrl_bounds=self.task.get_ctrl_bounds(),
                    max_steps=self.task.get_num_steps())
            else:
                surr_traj = simulate(controller, self.task.get_init_obs(),
                    ctrl_bounds=self.task.get_ctrl_bounds(),
                    term_cond=self.task.term_cond, sim_model=surrogate)
            cost = self.task.get_cost()
            surr_cost = cost(surr_traj)
            print("Surrogate Cost: ", surr_cost)
            print("Surrogate Final State: ", surr_traj[-1].obs)
            return surr_cost, (surr_traj.obs.tolist(), surr_traj.ctrls.tolist())
        except np.linalg.LinAlgError:
            return np.inf, None

    def __call__(self, controller):
        info = dict()
        info["surr_costs"] = []
        info["surr_trajs"] = []
        info["seeds"] = []
        for i, seed in enumerate(self.seeds):
            print("Simulating Surrogate Trajectory for Seed {}: ".format(i))
            surr_cost, surr_traj = self._run_sim(controller, seed)
            info["surr_costs"].append(surr_cost)
            info["surr_trajs"].append(surr_traj)

        distribution = NormalDistribution(np.mean(info["surr_costs"]), np.std(info["surr_costs"]))

        print("Surrogate Distribution: Mean {:.2f} Stddev {:.2f}".format(distribution.mu, distribution.sigma))

        return distribution, info

class OpenLoopController(Controller):
    def __init__(self, system, task, controls):
        super().__init__(system, task, model=None)
        self.controls = np.copy(controls)

    def traj_to_state(self, traj):
        return np.zeros(1, dtype=np.int64)

    @property
    def state_dim(self):
        return 1

    def run(self, state, new_obs):
        idx = state[0]
        return self.controls[idx,:], np.array([idx+1], dtype=np.int64)

    def is_compatible(self, *args, **kwargs):
        return True


class OpenLoopStochasticSurrogateEvaluator(SurrogateEvaluator):
    def __init__(self, system, task, trajs, surrogate_factory, rng, n_sims=10, surrogate_cfg=None,
                    surrogate_mode="defaultcfg", surrogate_tune_iters=100, holdout_prop=0.1):
        ControlEvaluator.__init__(self, system, task, trajs)
        self.surrogate_mode = surrogate_mode
        self.surrogate_factory = surrogate_factory
        self.seeds = [rng.integers(1 << 31) for _ in range(n_sims)]
        self.system = system
        self.task = task

        # Holdout evaluation data
        rng = np.random.default_rng(100)
        holdout_size = int(holdout_prop * len(trajs))
        shuffled_trajs = trajs[:]
        rng.shuffle(shuffled_trajs)
        self.holdout_trajs = shuffled_trajs[:holdout_size]
        self.surr_trajs = shuffled_trajs[holdout_size:]

        self.surrogate, self.surr_tune_result = self._get_surrogate(self.surr_trajs, rng, surrogate_tune_iters)

        residuals = get_model_residuals(self.surrogate, self.holdout_trajs, horizon=1)
        self.resid_mean = np.mean(residuals, axis=0)
        self.resid_cov  = np.cov(residuals.T)

    def _run_sim_orig(self, controller):
        surrogate = self.surrogate
        try:
            controller.reset()
            if self.task.has_num_steps():
                surr_traj = simulate(controller, self.task.get_init_obs(),
                    self.task.term_cond, sim_model=surrogate, 
                    ctrl_bounds=self.task.get_ctrl_bounds(),
                    max_steps=self.task.get_num_steps())
            else:
                surr_traj = simulate(controller, self.task.get_init_obs(),
                    ctrl_bounds=self.task.get_ctrl_bounds(),
                    term_cond=self.task.term_cond, sim_model=surrogate)
            cost = self.task.get_cost()
            surr_cost = cost(surr_traj)
            print("Surrogate Cost: ", surr_cost)
            print("Surrogate Final State: ", surr_traj[-1].obs)
            return surr_cost, surr_traj
        except np.linalg.LinAlgError:
            return np.inf, None

    def _run_sim_open_loop(self, orig_traj, seed):
        rng = np.random.default_rng(seed)
        surrogate = StochasticSurrogate(self.surrogate, self.resid_mean, self.resid_cov, rng)
        controller = OpenLoopController(self.system, self.task, orig_traj.ctrls)
        try:
            if self.task.has_num_steps():
                surr_traj = simulate(controller, self.task.get_init_obs(),
                    self.task.term_cond, sim_model=surrogate, 
                    ctrl_bounds=self.task.get_ctrl_bounds(),
                    max_steps=self.task.get_num_steps())
            else:
                surr_traj = simulate(controller, self.task.get_init_obs(),
                    ctrl_bounds=self.task.get_ctrl_bounds(),
                    term_cond=self.task.term_cond, sim_model=surrogate)
            cost = self.task.get_cost()
            surr_cost = cost(surr_traj)
            print("Surrogate Cost: ", surr_cost)
            print("Surrogate Final State: ", surr_traj[-1].obs)
            return surr_cost,  (surr_traj.obs.tolist(), surr_traj.ctrls.tolist())
        except np.linalg.LinAlgError:
            return np.inf, None

    def __call__(self, controller):
        info = dict()
        info["surr_costs"] = []
        info["surr_trajs"] = []
        info["seeds"] = []
        orig_cost, orig_traj = self._run_sim_orig(controller)
        info["orig_cost"] = orig_cost
        info["orig_traj"] = (orig_traj.obs.tolist(), orig_traj.ctrls.tolist())
        for i, seed in enumerate(self.seeds):
            print("Simulating Surrogate Trajectory for Seed {}: ".format(i))
            surr_cost, surr_traj = self._run_sim_open_loop(orig_traj, seed)
            info["surr_costs"].append(surr_cost)
            info["surr_trajs"].append(surr_traj)

        distribution = NormalDistribution(np.mean(info["surr_costs"]), np.std(info["surr_costs"]))

        print("Surrogate Distribution: Mean {:.2f} Stddev {:.2f}".format(distribution.mu, distribution.sigma))

        return distribution, info