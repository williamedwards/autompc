# Created by Teodor Tchalakov, (ttcha2@illinoi.edu)

import numpy as np
import numpy.linalg as la

from .cost import Cost

class LogBarrierCost(Cost):
    def __init__(self, system, observation, limit, scale=1, upper=True):#TODO: Add controls cost
        """
        Create barrier cost that approximates an inequality constraint.
        where : - b * ln ( a - x ) for upper limit
                - b * ln ( a + x ) for lower limit

        Parameters
        ----------
        system : System
            Robot system object.

        observation : String
            Observation name for which limit is specified.

        limit : Numpy array
            limit value a that barrier is placed at.

        scale : double
            Positive scaler to magnify the cost function.
            scale: (0, inf)

        upper : boolean
            True if the limit is an upper limit.
            False if the limit is a lower limit
        """
        super().__init__(system)
        self._obs_id = system.observations.index(observation)
        self._limit = np.copy(limit)
        self._direction = -1
        self._scale = scale
        if(upper):
            self.direction = 1
        

        # Configs
        self._is_quad = False
        self._is_convex = False     #TODO: Not sure
        self._is_diff = True
        self._is_twice_diff = True
        self._has_goal = False      #TODO: Probably not

    #Cost Function:
    # - b * ln ( a - x ) upper limit
    # - b * ln ( a + x ) lower limit
    def eval_obs_cost(self, obs):
        return -self._scale * np.log( self._limit - (self._direction * obs[self._obs_id]))

    #Jacobian:
    # b / (a - x) upper limit
    # -b / (a - x) lower limit
    def eval_obs_cost_diff(self, obs):
        return self.direction * self._scale / (self._limit - obs[self._obs_id])

    #Hessian:
    # b / (a - x)^2 upper limit
    # b / (a - x)^2 lower limit
    def eval_obs_cost_hess(self, obs):
        return self._scale / ((self._limit - obs[self._obs_id])**2)

    def eval_ctrl_cost(self, ctrl):
        return 0
    
    def eval_ctrl_cost_diff(self, ctrl):
        return 0

    def eval_ctrl_cost_hess(self, ctrl):
        return 0

    def eval_term_obs_cost(self, obs):
        return 0

    def eval_term_obs_cost_diff(self, obs):
        return 0

    def eval_term_obs_cost_hess(self, obs):
        return 0