# Created by Teodor Tchalakov, (ttcha2@illinoi.edu)

import numpy as np
import numpy.linalg as la

from .cost import Cost

class LogBarrierCost(Cost):
    def __init__(self, system, boundedState):#TODO: Add controls cost
        """
        Create barrier cost that approximates an inequality constraint.
        where : - b * ln ( a - x ) for upper limit
                - b * ln ( a + x ) for lower limit

        Parameters
        ----------
        system : System
            Robot system object.

        boundedState : dict
            Dictionary of { "observation or control name" : (limit, scale, upper)}

            observation : String
                Observation name for which limit is specified.

            limit : double
                limit value a that barrier is placed at.

            scale : double
                Positive scaler to magnify the cost function.
                scale: (0, inf)

            upper : boolean
                True if the limit is an upper limit.
                False if the limit is a lower limit
        """
        super().__init__(system)
        self.obsConfiguration = []
        self.ctrlConfiguration = []

        for variable in boundedState.keys():
            if(variable in system.observations):
                obs_id = system.observations.index(variable)
                self.obsConfiguration.append([obs_id, boundedState[variable]])
            if(variable in system.controls):
                obs_id = system.controls.index(variable)
                self.obsConfiguration.append([obs_id, boundedState[variable]])
        
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
        sum = 0
        for boundedObs in self.obsConfiguration:
            limit, scale, upper = boundedObs[1]
            self._direction = -1
            if(upper):
                direction = 1
            sum = sum + -scale * np.log(limit - (direction * obs[boundedObs[0]]))
        return sum

    #Jacobian:
    # b / (a - x) upper limit
    # -b / (a - x) lower limit
    def eval_obs_cost_diff(self, obs):
        jacobian = np.zeros(self.system.obs_dim)
        for boundedObs in self.obsConfiguration:
            limit, scale, upper = boundedObs[1]
            self._direction = -1
            if(upper):
                direction = 1
            jacobian[boundedObs[0]] = direction * scale / (limit - obs[boundedObs[0]])
        return jacobian

    #Hessian:
    # b / (a - x)^2 upper limit
    # b / (a - x)^2 lower limit
    def eval_obs_cost_hess(self, obs):
        hessian = np.zeros((self.system.obs_dim, self.system.obs_dim))
        for boundedObs in self.obsConfiguration:
            limit, scale, upper = boundedObs[1]
            hessian[boundedObs[0]][boundedObs[0]] = scale / ((limit - obs[boundedObs[0]])**2)
        return hessian

    def eval_ctrl_cost(self, ctrl):
        sum = 0
        for boundedCtrl in self.ctrlConfiguration:
            limit, scale, upper = boundedCtrl[1]
            self._direction = -1
            if(upper):
                direction = 1
            sum = sum + -scale * np.log(limit - (direction * ctrl[boundedCtrl[0]]))
        return sum
    
    def eval_ctrl_cost_diff(self, ctrl):
        jacobian = np.zeros(self.system.ctrl_dim)
        for boundedCtrl in self.obsConfiguration:
            limit, scale, upper = boundedCtrl[1]
            self._direction = -1
            if(upper):
                direction = 1
            jacobian[boundedCtrl[0]] = direction * scale / (limit - ctrl[boundedCtrl[0]])
        return jacobian

    def eval_ctrl_cost_hess(self, ctrl):
        hessian = np.zeros((self.system.obs_dim, self.system.obs_dim))
        for boundedCtrl in self.obsConfiguration:
            limit, scale, upper = boundedCtrl[1]
            hessian[boundedCtrl[0]][boundedCtrl[0]] = scale / ((limit - ctrl[boundedCtrl[0]])**2)
        return hessian

    def eval_term_obs_cost(self, obs):
        return 0

    def eval_term_obs_cost_diff(self, obs):
        return 0

    def eval_term_obs_cost_hess(self, obs):
        return 0