import numpy as np
from .control_evaluator import ControlEvaluator, ConstantDistribution
from ..utils import simulate

class TrueDynamicsEvaluator(ControlEvaluator):
    def __init__(self, system, tasks, trajs, dynamics):
        super().__init__(system, tasks, trajs)
        self.dynamics = dynamics

    def _evaluate_for_task(self, controller, task):
        info = dict()
        print("Simulating True Dynamics Trajectory: ")
        try:
            controller.set_ocp(task.get_ocp())
            controller.reset()
            if task.has_num_steps():
                truedyn_traj = simulate(controller, task.get_init_obs(),
                    task.term_cond, dynamics=self.dynamics, 
                    ctrl_bounds=task.get_ocp().get_ctrl_bounds(),
                    max_steps=task.get_num_steps())
            else:
                truedyn_traj = simulate(controller, task.get_init_obs(),
                    ctrl_bounds=task.get_ctrl_bounds(),
                    term_cond=task.term_cond, dynamics=self.dynamics)
            cost = task.get_ocp().get_cost()
            truedyn_cost = cost(truedyn_traj)
            print("True Dynamics Cost: ", truedyn_cost)
            print("True Dynamics Final State: ", truedyn_traj[-1].obs)
            info["truedyn_cost"] = truedyn_cost
            info["truedyn_traj"] = (truedyn_traj.obs.tolist(), truedyn_traj.ctrls.tolist())
        except np.linalg.LinAlgError:
            truedyn_cost = np.inf
            info["truedyn_cost"] = truedyn_cost
            info["truedyn_traj"] = None

        return truedyn_cost, info

    def __call__(self, controller):
        if len(self.tasks) == 1:
            truedyn_cost, info = self._evaluate_for_task(controller, self.tasks[0])
            distribution = ConstantDistribution(truedyn_cost)
            return distribution, info
        else:
            info = {"task_infos" : []}
            costs = []
            for i, task in enumerate(self.tasks):
                print(f"Evaluating Task {i}")
                task_cost, task_info = self._evaluate_for_task(controller, task)
                info["task_infos"].append(task_info)
                costs.append(task_cost)
            distribution = ConstantDistribution(np.mean(costs))
            return distribution, info