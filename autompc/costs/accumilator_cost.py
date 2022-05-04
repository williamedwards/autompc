# Created by Teodor Tchalakov, (ttcha2@illinois.edu)

import numpy as np
import numpy.linalg as la

from .cost import Cost

class AccumilatorCost(Cost):
    def __init__(self, system, obs_cost=None, ctrl_cost=None):
        """
        Create accumilator cost where at each timestep of a trajectory a
        control/observation cost is computed and summed.

        Note:
        It is not recommended that users keep track of state history in defined functions.

        Parameters
        ----------
        system : System
            Robot system object

        obs_cost: Function obs, control -> cost
            numpy array of observations of shape (system.obs_dim) -> float cost

        ctrl_cost: Function obs, control -> cost
            numpy array of controls of shape (system.ctrl_dim) -> float cost
        """
        super().__init__(system)
        if obs_cost is None and ctrl_cost is None:
            raise ValueError("Accumilator Cost requires alteast an observation or control cost function")
        self._obs_cost = obs_cost
        self._ctrl_cost = ctrl_cost

        self._is_quad = False
        self._is_convex = False
        self._is_diff = False
        self._is_twice_diff = False
        self._has_goal = False

    def eval_obs_cost(self, obs):
        if self._ctrl_cost is None:
            return 0
        else:
            return self._obs_cost(obs)

    def eval_ctrl_cost(self, ctrl):
        if self._ctrl_cost is None:
            return 0
        else:
            return self._ctrl_cost(ctrl)

    def eval_term_obs_cost(self, obs):
        return 0.0