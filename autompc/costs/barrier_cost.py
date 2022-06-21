# Created by Teodor Tchalakov, (ttcha2@illinois.edu)

import numpy as np
import numpy.linalg as la

from .cost import Cost

class LogBarrierCost(Cost):
    def __init__(self, system, boundedStates):
        """
        Create barrier cost that approximates an inequality constraint.
        Function does not exist outside the limit.
        where : - b * ln ( a - x ) for upper limit
                - b * ln ( a + x ) for lower limit
        Parameters
        ----------
        system : System
            Robot system object.
        boundedState : dict
            Dictionary of { "observation/control name" : (limit, scale, upper)}
                observation/control (x) : String
                    Observation/control name for which limit is specified.
                limit (a) : double
                    limit value a that barrier is placed at.
                scale (b) : double
                    Positive scalar to magnify the cost function.
                    scale: (0, inf)
                upper : boolean
                    True if the limit is an upper limit.
                    False if the limit is a lower limit.
        """
        super().__init__(system)
        self.obsConfiguration = []
        self.ctrlsConfiguration = []

        for variable in boundedStates.keys():
            config = boundedStates[variable]
            # Check that scale is positive
            if(config[1] < 0):
                raise ValueError(f"{variable}'s log barrier must be positive, was {config[1]}")
            elif(variable in system.observations):
                self.obsConfiguration.append([variable, config])
            elif(variable in system.controls):
                self.ctrlsConfiguration.append([variable, config])
            else:
                raise ValueError(f"Variable {variable} is not in the given system")
        
        # Configs
        self._is_quad = False
        self._is_convex = True
        self._is_diff = True
        self._is_twice_diff = True
        self._has_goal = False

    #Cost Function:
    # - b * ln ( a - x ) upper limit
    # - b * ln ( a + x ) lower limit
    def eval_obs_cost(self, obs):
        sum = 0
        for boundedObs in self.obsConfiguration:
            variable, config = boundedObs
            index = self.system.observations.index(variable)
            limit, scale, upper = config
            self._direction = -1
            if(upper):
                direction = 1
            sum = sum + -scale * np.log(limit - (direction * obs[index]))
        return sum

    #Jacobian:
    # b / (a - x) upper limit
    # -b / (a - x) lower limit
    def eval_obs_cost_diff(self, obs):
        jacobian = np.zeros(self.system.obs_dim)
        for boundedObs in self.obsConfiguration:
            variable, config = boundedObs
            index = self.system.observations.index(variable)
            limit, scale, upper = config
            self._direction = -1
            if(upper):
                direction = 1
            jacobian[boundedObs[0]] = direction * scale / (limit - obs[index])
        return self.eval_obs_cost(obs), jacobian

    #Hessian:
    # b / (a - x)^2 upper limit
    # b / (a - x)^2 lower limit
    def eval_obs_cost_hess(self, obs):
        hessian = np.zeros((self.system.obs_dim, self.system.obs_dim))
        for boundedObs in self.obsConfiguration:
            variable, config = boundedObs
            index = self.system.observations.index(variable)
            limit, scale, _ = config
            hessian[boundedObs[0]][boundedObs[0]] = scale / ((limit - obs[index])**2)
        return *self.eval_obs_cost_diff, hessian

    def eval_ctrl_cost(self, ctrl):
        sum = 0
        for boundedCtrl in self.ctrlsConfiguration:
            variable, config = boundedCtrl
            index = self.system.controls.index(variable)
            limit, scale, upper = config
            self._direction = -1
            if(upper):
                direction = 1
            sum = sum + -scale * np.log(limit - (direction * ctrl[index]))
        return sum
    
    def eval_ctrl_cost_diff(self, ctrl):
        jacobian = np.zeros(self.system.ctrl_dim)
        for boundedCtrl in self.ctrlsConfiguration:
            variable, config = boundedCtrl
            index = self.system.controls.index(variable)
            limit, scale, upper = config
            self._direction = -1
            if(upper):
                direction = 1
            jacobian[boundedCtrl[0]] = direction * scale / (limit - ctrl[index])
        return self.eval_ctrl_cost(), jacobian

    def eval_ctrl_cost_hess(self, ctrl):
        hessian = np.zeros((self.system.ctrl_dim, self.system.ctrl_dim))
        for boundedCtrl in self.ctrlsConfiguration:
            variable, config = boundedCtrl
            index = self.system.controls.index(variable)
            limit, scale, _ = config
            hessian[boundedCtrl[0]][boundedCtrl[0]] = scale / ((limit - ctrl[index])**2)
        return *self.eval_ctrl_cost_diff(), hessian

    def eval_term_obs_cost(self, obs):
        return 0

    def eval_term_obs_cost_diff(self, obs):
        return 0

    def eval_term_obs_cost_hess(self, obs):
        return 0