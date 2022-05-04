# Created by Teodor Tchalakov, (ttcha2@illinois.edu)

import numpy as np
import numpy.linalg as la

from .cost import Cost

class DistanceCost(Cost):
    def __init__(self, system, goal_state, obs_range=None, observations=None):
        """
        Create difference cost. Returns euclidean(L2) norm of an observation
        to the goal observation for every time step
        The check is performed only over the observation dimensions from
        obs_range[0] to obs_range[1].

        Note:
        Can be used as a metric for accuracy.

        Parameters
        ----------
        system : System
            Robot system object

        goal : Numpy array
            Goal position

        obs_range : (int, int)
            First (inclusive and last (exclusive) index of observations
            for which goal is specified.  If neither this field nor
            observations is set, default is full observation range.

        observations : [str]
            List of observation names for which goal is specified.
            Supersedes obs_range when present.
        """
        super().__init__(system)
        self._obs_idxs = None
        if obs_range is not None:
            self._obs_idxs = list(range(obs_range[0], obs_range[1]))
        if observations is not None:
            self._obs_idxs = [system.observations.index(obs) for obs in observations]
        if self._obs_idxs is None:
            self._obs_idxs = list(range(0, system.obs_dim))
        self.set_goal(goal_state)

        self._is_quad = False
        self._is_convex = False
        self._is_diff = False
        self._is_twice_diff = False

        if goal_state is None:
            self._has_goal = False
        else:
            self._goal = np.copy(goal_state)
            self._has_goal = True

    def set_goal(self, goal):
        if len(goal) < self.system.obs_dim:
            self._goal = np.zeros(self.system.obs_dim)
            self._goal[self._obs_idxs] = goal
        else:
            self._goal = np.copy(goal)

    def eval_obs_cost(self, obs):
        return la.norm(obs[self._obs_idxs] - self._goal[self._obs_idxs], np.inf)

    def eval_ctrl_cost(self, ctrl):
        return 0.0

    def eval_term_obs_cost(self, obs):
        return 0.0