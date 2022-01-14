# Created by William Edwards (wre2@illinois.edu), 2021-12-22

import numpy as np

from .task import Task

class StaticGoalTask(Task):
    def __init__(self, system):
        super().__init__(system)
        self._parameter_names.append("goal")

    def set_goal(self, goal):
        goal = np.array(goal)
        if not goal.shape == (self.system.obs_dim,):
            raise ValueError("goal is not correct shape")
        self.set_parameter("goal", goal)

    def get_goal(self):
        return self.get_parameter("goal")