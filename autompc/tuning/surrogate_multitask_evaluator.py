import numpy as np
from .control_evaluator import ControlEvaluator, ConstantDistribution
from .surrogate_evaluator import SurrogateEvaluator
from ..utils import simulate

class SurrogateMultiTaskEvaluator(SurrogateEvaluator):
    def __init__(self, system, task, trajs, *args, **kwargs):
        super().__init__(system, task, trajs, *args, **kwargs)

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
                    term_cond=self.task.term_cond, sim_model=self.surrogate)
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

        return distribution, info
        
