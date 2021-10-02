import numpy as np
from .control_evaluator import ControlEvaluator, ConstantDistribution
from ..utils import simulate

class TrueDynamicsEvaluator(ControlEvaluator):
    def __init__(self, system, task, trajs, dynamics):
        super().__init__(system, task, trajs)
        self.dynamics = dynamics

    def __call__(self, controller):
        info = dict()
        print("Simulating True Dynamics Trajectory")
        controller.reset()
        if self.task.has_num_steps():
            truedyn_traj = simulate(controller, self.task.get_init_obs(),
                self.task.term_cond, dynamics=self.dynamics, max_steps=self.task.get_num_steps())
        else:
            truedyn_traj = simulate(controller, self.task.get_init_obs(),
                self.task.term_cond, dynamics=self.dynamics)
        cost = self.task.get_cost()
        truedyn_cost = cost(truedyn_traj)
        print("True Dynamics Cost: ", truedyn_cost)
        print("True Dynamics Final State: ", truedyn_traj[-1].obs)
        info["truedyn_cost"] = truedyn_cost
        info["truedyn_traj"] = (truedyn_traj.obs.tolist(), 
                truedyn_traj.ctrls.tolist())

        distribution = ConstantDistribution(truedyn_cost)

        return distribution, info

