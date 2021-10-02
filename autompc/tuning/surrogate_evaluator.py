import numpy as np
from .control_evaluator import ControlEvaluator, ConstantDistribution
from ..utils import simulate

class SurrogateEvaluator(ControlEvaluator):
    def __init__(self, system, task, trajs, surrogate_factory, rng=None, surrogate_cfg=None,
                    surrogate_mode="defaultcfg", surrogate_tune_iters=100):
        super().__init__(system, task, trajs)
        self.surr_trajs = trajs
        self.surrogate_mode = surrogate_mode
        self.surrogate_factory = surrogate_factory
        self.surrogate_cfg = surrogate_cfg

        self.surrogate, self.surr_tune_result = self._get_surrogate(self.surr_trajs, rng, surrogate_tune_iters)

    def _get_surrogate(self, trajs, rng, surrogate_tune_iters):
        surrogate_tune_result = None
        if self.surrogate_mode == "defaultcfg":
            surrogate_cs = self.surrogate_factory.get_configuration_space()
            surrogate_cfg = surrogate_cs.get_default_configuration()
            surrogate = self.surrogate_factory(surrogate_cfg, trajs)
        elif self.surrogate_mode == "fixedcfg":
            surrogate = self.surrogate_factory(self.surrogate_cfg, trajs)
        elif self.surrogate_mode == "autotune":
            evaluator = self.surrogate_evaluator
            if evaluator is None:
                evaluator = HoldoutModelEvaluator(
                        holdout_prop = self.surrogate_tune_holdout,
                        metric = self.surrogate_tune_metric,
                        system=self.system,
                        trajs=trajs,
                        rng=rng)
            model_tuner = ModelTuner(self.system, evaluator) 
            model_tuner.add_model_factory(self.surrogate_factory)
            surrogate, surrogate_tune_result = model_tuner.run(rng, n_iters=surrogate_tune_iters) 
        elif self.surrogate_mode == "autoselect":
            evaluator = self.surrogate_evaluator
            if evaluator is None:
                evaluator = HoldoutModelEvaluator(
                        holdout_prop = self.surrogate_tune_holdout,
                        metric = self.surrogate_tune_metric,
                        system=self.system,
                        trajs=trajs,
                        rng=rng)
            model_tuner = ModelTuner(self.system, evaluator) 
            for factory in autoselect_factories:
                model_tuner.add_model_factory(factory(self.system))
            surrogate, surrogate_tune_result = model_tuner.run(rng, n_iters=surrogate_tune_iters) 
        return surrogate, surrogate_tune_result

    def __call__(self, controller):
        info = dict()
        print("Simulating Surrogate Trajectory: ")
        try:
            controller.reset()
            if self.task.has_num_steps():
                surr_traj = simulate(controller, self.task.get_init_obs(),
                    self.task.term_cond, sim_model=self.surrogate, 
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
            info["surr_cost"] = surr_cost
            info["surr_traj"] = (surr_traj.obs.tolist(), surr_traj.ctrls.tolist())
        except np.linalg.LinAlgError:
            surr_cost = np.inf
            info["surr_cost"] = surr_cost
            info["surr_traj"] = None

        # Return constant distribution
        distribution = ConstantDistribution(surr_cost)

        return distribution, surr_cost
        