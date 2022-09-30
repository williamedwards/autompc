# Created by Teodor Tchalakov, (ttcha2@illinois.edu)

import numpy as np
import numpy.linalg as la

from .cost import Cost

class LogBarrierCost(Cost):
    def __init__(self, system, obs_bounds, ctrl_bounds, scales):
        """
        Create barrier cost that approximates an inequality constraint.
        Function does not exist outside the limit.
        where : - b * ln ( a - x ) for upper limit
                - b * ln ( a + x ) for lower limit
        Parameters
        ----------
        system : System
            Robot system object.
        scales : dict
            Dictionary of { "observation/control name" : scale}
                observation/control (x) : String
                    Observation/control name for which barrier is placed.
                scale (b) : double
                    Positive scalar to magnify the cost function.
                    scale: (0, inf)
        """
        super().__init__(system)
        self.obs_bounds = obs_bounds
        self.ctrl_bounds = ctrl_bounds
        self.scales = scales

        self.obsConfiguration = []
        self.ctrlsConfiguration = []

        for variable in scales.keys():
            scale = scales[variable]
            # Check that scale is positive
            if(scale < 0):
                raise ValueError(f"{variable}'s log barrier scale must be positive, was {scale}")
            elif scale == 0:
                # Skip constructing barrier since transformer scale 0
                continue
            elif(variable in system.observations):
                lower, upper = obs_bounds[self.system.observations.index(variable)]
                self.obsConfiguration.append([variable, (lower, upper, scale)])
            elif(variable in system.controls):
                lower, upper = ctrl_bounds[self.system.controls.index(variable)]
                self.ctrlsConfiguration.append([variable, (lower, upper, scale)])
            else:
                raise ValueError(f"Variable {variable} is not in the given system")
        
        # Configs
        self._is_quad = False
        self._is_convex = True
        self._is_diff = True
        self._is_twice_diff = True
        self._has_goal = False

    def incremental(self, obs, control):
        return self.eval_obs_cost(obs) + self.eval_ctrl_cost(control)

    def incremental_diff(self, obs, control):
        return self.incremental(obs, control), self.eval_obs_cost_diff(obs), self.eval_ctrl_cost_diff(control)

    def incremental_hess(self, obs, control): # TODO: Tuple unpacking only supported for python>=3.8
        hess_obs_ctrl = np.zeros((self.system.obs_dim, self.system.ctrl_dim))
        return self.incremental(obs, control), self.eval_obs_cost_diff(obs), self.eval_ctrl_cost_diff(control), self.eval_obs_cost_hess(obs), hess_obs_ctrl, self.eval_ctrl_cost_hess(control)

    def terminal(self, obs):
        return 0

    def terminal_diff(self, obs):
        return 0, np.zeros(self.system.obs_dim)
    
    def terminal_hess(self, obs):
        return 0, np.zeros(self.system.obs_dim), np.zeros((self.system.obs_dim, self.system.obs_dim))

    def __add__(self, rhs):
        if isinstance(rhs, LogBarrierCost):
            if (self.goal is None and rhs.goal is None) or np.all(self.goal == rhs.goal):
                return LogBarrierCost(self.system, self.obs_bounds, self.ctrl_bounds, self.scales+rhs.scales)
        return Cost.__add__(self, rhs)

    def __mul__(self, rhs):
        if not isinstance(rhs, (float, int)):
            raise ValueError("* only supports product with numbers")
        new_cost = LogBarrierCost(self.system, self.obs_bounds, self.ctrl_bounds, self.scales)
        return new_cost


    #Cost Function:
    # b = scale
    # - b * ln ( a - x ) upper limit x < a
    # - b * ln ( -a + x ) lower limit x > a := -x < -a
    def eval_obs_cost(self, obs):
        sum = 0
        for boundedObs in self.obsConfiguration:
            variable, config = boundedObs
            index = self.system.observations.index(variable)
            lower, upper, scale = config
            if lower > -np.inf:
                if lower >= obs[index]:
                    sum += np.inf
                else:
                    sum = sum + -scale * np.log(-lower + obs[index])
            if upper < np.inf:
                if obs[index] >= upper:
                    sum += np.inf
                else:
                    sum = sum + -scale * np.log(upper - obs[index])           
        return sum

    #Jacobian:
    # b / (a - x) upper limit
    # -b / (-a + x) lower limit
    def eval_obs_cost_diff(self, obs):
        jacobian = np.zeros(self.system.obs_dim)
        for boundedObs in self.obsConfiguration:
            variable, config = boundedObs
            index = self.system.observations.index(variable)
            lower, upper, scale = config
            if lower > -np.inf:
                if lower >= obs[index]:
                    jacobian[index] += -np.inf
                else:
                    jacobian[index] += -scale / (-lower + obs[index])
            if upper < np.inf:
                if obs[index] >= upper:
                    jacobian[index] += np.inf
                else:
                    jacobian[index] += scale / (upper - obs[index])   
            
        return jacobian

    #Hessian:
    # b / (a - x)^2 upper limit
    # b / (-a + x)^2 lower limit
    def eval_obs_cost_hess(self, obs):
        hessian = np.zeros((self.system.obs_dim, self.system.obs_dim))
        for boundedObs in self.obsConfiguration:
            variable, config = boundedObs
            index = self.system.observations.index(variable)
            lower, upper, scale = config
            if lower > -np.inf:
                if lower >= obs[index]:
                    hessian[index][index] += np.inf
                else:
                    hessian[index][index] += scale / ((lower - obs[index])**2)
            if upper < np.inf:
                if obs[index] >= upper:
                    hessian[index][index] += np.inf
                else:
                    hessian[index][index] += scale / ((upper - obs[index])**2)
            
        return hessian

    def eval_ctrl_cost(self, ctrl):
        sum = 0
        for boundedCtrl in self.ctrlsConfiguration:
            variable, config = boundedCtrl
            index = self.system.controls.index(variable)
            lower, upper, scale = config
            if lower > -np.inf:
                sum = sum + -scale * np.log(-lower + ctrl[index])
            if upper < np.inf:
                sum = sum + -scale * np.log(upper - ctrl[index])
        return sum
    
    def eval_ctrl_cost_diff(self, ctrl):
        jacobian = np.zeros(self.system.ctrl_dim)
        for boundedCtrl in self.ctrlsConfiguration:
            variable, config = boundedCtrl
            index = self.system.controls.index(variable)
            lower, upper, scale = config
            if lower > -np.inf:
                jacobian[index] += -scale / (-lower + ctrl[index])
            if upper < np.inf:
                jacobian[index] += scale / (upper - ctrl[index])   
        return jacobian

    def eval_ctrl_cost_hess(self, ctrl):
        hessian = np.zeros((self.system.ctrl_dim, self.system.ctrl_dim))
        for boundedCtrl in self.ctrlsConfiguration:
            variable, config = boundedCtrl
            index = self.system.controls.index(variable)
            lower, upper, scale = config
            if lower > -np.inf:
                hessian[index][index] += scale / ((lower - ctrl[index])**2)
            if upper < np.inf:
                hessian[index][index] += scale / ((upper - ctrl[index])**2)
        return hessian