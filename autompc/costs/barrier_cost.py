# Created by Teodor Tchalakov, (ttcha2@illinois.edu)

import numpy as np
import numpy.linalg as la

from .cost import Cost

class BarrierCost(Cost):
    def __init__(self, system, obs_bounds, ctrl_bounds, scales):
        """
        Abstract Class that encompasses both soft and hard barrier costs. 
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

        for variable, scale in scales.items():
            # Check that scale is positive
            if(scale < 0):
                raise ValueError(f"{variable}'s barrier scale must be positive, was {scale}")
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
    
    @property
    def is_hard(self) -> bool:
        """
        True if cost is a hard barrier
        """
        return NotImplementedError

    def barrier(self, value, lower, upper) -> float:
        return NotImplementedError
        
    def barrier_diff(self, value, lower, upper) -> float:
        return NotImplementedError

    def barrier_hess(self, value, lower, upper) -> float:
        return NotImplementedError
    
    def incremental(self, obs, control):
        sum = 0
        # Observation cost
        for boundedObs in self.obsConfiguration:
            variable, config = boundedObs
            index = self.system.observations.index(variable)
            lower, upper, scale = config
            lower_barrier_cost, upper_barrier_cost = self.barrier(obs[index], lower, upper)
            if lower > -np.inf:
                if lower >= obs[index]:
                    if self.is_hard:
                        sum += np.inf
                    else:
                        sum += scale * lower_barrier_cost
                else:
                    if self.is_hard:
                        sum += scale * lower_barrier_cost
            if upper < np.inf:
                if obs[index] >= upper:
                    if self.is_hard:
                        sum += np.inf
                    else:
                        sum += scale * upper_barrier_cost
                else:
                    if self.is_hard:
                        sum += scale * upper_barrier_cost

        # Control cost
        for boundedCtrl in self.ctrlsConfiguration:
            variable, config = boundedCtrl
            index = self.system.controls.index(variable)
            lower, upper, scale = config
            lower_barrier_cost, upper_barrier_cost = self.barrier(control[index], lower, upper)
            if lower > -np.inf:
                if lower >= obs[index]:
                    if self.is_hard:
                        sum += np.inf
                    else:
                        sum += scale * lower_barrier_cost
                else:
                    if self.is_hard:
                        sum += scale * lower_barrier_cost
            if upper < np.inf:
                if obs[index] >= upper:
                    if self.is_hard:
                        sum += np.inf
                    else:
                        sum += scale * upper_barrier_cost
                else:
                    if self.is_hard:
                        sum += scale * upper_barrier_cost
        return sum

    def incremental_diff(self, obs, control):
        obs_jacobian = np.zeros(self.system.obs_dim)
        for boundedObs in self.obsConfiguration:
            variable, config = boundedObs
            index = self.system.observations.index(variable)
            lower, upper, scale = config
            lower_barrier_diff, upper_barrier_diff = self.barrier_diff(obs[index], lower, upper)
            if lower > -np.inf:
                if lower >= obs[index]:
                    if self.is_hard:
                        obs_jacobian[index] += -np.inf
                    else:
                        obs_jacobian[index] += scale * lower_barrier_diff
                else:
                    if self.is_hard:
                        obs_jacobian[index] += scale * lower_barrier_diff
            if upper < np.inf:
                if obs[index] >= upper:
                    if self.is_hard:
                        obs_jacobian[index] += np.inf
                    else:
                        obs_jacobian[index] += scale * upper_barrier_diff
                else:
                    if self.is_hard:
                        obs_jacobian[index] += scale * upper_barrier_diff

        ctrl_jacobian = np.zeros(self.system.obs_dim)
        for boundedCtrl in self.ctrlsConfiguration:
            variable, config = boundedCtrl
            index = self.system.controls.index(variable)
            lower, upper, scale = config
            lower_barrier_diff, upper_barrier_diff = self.barrier_diff(control[index], lower, upper)
            if lower > -np.inf:
                if lower >= obs[index]:
                    if self.is_hard:
                        ctrl_jacobian[index] += -np.inf
                    else:
                        ctrl_jacobian[index] += scale * lower_barrier_diff
                else:
                    if self.is_hard:
                        ctrl_jacobian[index] += scale * lower_barrier_diff
            if upper < np.inf:
                if obs[index] >= upper:
                    if self.is_hard:
                        ctrl_jacobian[index] += np.inf
                    else:
                        ctrl_jacobian[index] += scale * upper_barrier_diff
                else:
                    if self.is_hard:
                        ctrl_jacobian[index] += scale * upper_barrier_diff

        return self.incremental(obs, control), obs_jacobian, ctrl_jacobian

    def incremental_hess(self, obs, control): 
        obs_hessian = np.zeros((self.system.obs_dim, self.system.obs_dim))
        for boundedObs in self.obsConfiguration:
            variable, config = boundedObs
            index = self.system.observations.index(variable)
            lower, upper, scale = config
            lower_barrier_hess, upper_barrier_hess = self.barrier_hess(obs[index], lower, upper)
            if lower > -np.inf:
                if lower >= obs[index]:
                    if self.is_hard:
                        obs_hessian[index][index] += np.inf
                    else:
                        obs_hessian[index][index] += scale * lower_barrier_hess
                else:
                    if self.is_hard:
                        obs_hessian[index][index] += scale * lower_barrier_hess
            if upper < np.inf:
                if obs[index] >= upper:
                    if self.is_hard:
                        obs_hessian[index][index] += np.inf
                    else:
                        obs_hessian[index][index] += scale * upper_barrier_hess
                else:
                    if self.is_hard:
                        obs_hessian[index][index] += scale * upper_barrier_hess
            

        obs_ctrl_hessian = np.zeros((self.system.obs_dim, self.system.ctrl_dim))

        ctrl_hessian = np.zeros((self.system.obs_dim, self.system.obs_dim))
        for boundedCtrl in self.ctrlsConfiguration:
            variable, config = boundedCtrl
            index = self.system.controls.index(variable)
            lower, upper, scale = config
            lower_barrier_hess, upper_barrier_hess = self.barrier_hess(control[index], lower, upper)
            if lower > -np.inf:
                if lower >= obs[index]:
                    if self.is_hard:
                        ctrl_hessian[index][index] += np.inf
                    else:
                        ctrl_hessian[index][index] += scale * lower_barrier_hess
                else:
                    if self.is_hard:
                        ctrl_hessian[index][index] += scale * lower_barrier_hess
            if upper < np.inf:
                if obs[index] >= upper:
                    if self.is_hard:
                        ctrl_hessian[index][index] += np.inf
                    else:
                        ctrl_hessian[index][index] += scale * upper_barrier_hess
                else:
                    if self.is_hard:
                        ctrl_hessian[index][index] += scale * upper_barrier_hess
        return self.incremental(obs, control), *self.incremental_diff(obs, control), obs_hessian, obs_ctrl_hessian, ctrl_hessian

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


class LogBarrierCost(BarrierCost):
    @property
    def is_quad(self) -> bool:
        return False

    @property
    def is_convex(self) -> bool:
        return True

    @property
    def is_diff(self) -> bool:
        return True

    @property
    def is_twice_diff(self) -> bool:
        return True

    @ property
    def has_goal(self) -> bool:
        return False

    @property
    def is_hard(self) -> bool:
        return True
    
    def barrier(self, value, lower, upper):
        """
        Cost Function:
        b = scale
        b * -ln ( -a + x ) upper limit x < a
        b * -ln (  a - x ) lower limit x > a := -x < -a"""
        return -np.log(-lower + value),  -np.log(upper - value)

    def barrier_diff(self, value, lower, upper):
        """
        Jacobian:
        b / (a - x) upper limit
        b / (a - x) lower limit
        """
        return 1 / (lower - value), 1 / (upper - value)

    def barrier_hess(self, value, lower, upper):
        """
        Hessian:
        b / (a - x)^2 upper limit
        b / (a - x)^2 lower limit
        """
        return 1 / ((lower - value)**2), 1 / ((upper - value)**2)

class InverseBarrierCost(BarrierCost):
    @property
    def is_quad(self) -> bool:
        return False

    @property
    def is_convex(self) -> bool:
        return True

    @property
    def is_diff(self) -> bool:
        return True

    @property
    def is_twice_diff(self) -> bool:
        return True

    @ property
    def has_goal(self) -> bool:
        return False

    @property
    def is_hard(self) -> bool:
        return True
    
    def barrier(self, value, lower, upper):
        """
        Cost Function:
        b = scale
        b * 1 / ( -a + x ) upper limit x < a
        b * 1 / (  a - x ) lower limit x > a := -x < -a"""
        return 1/(-lower + value),  1/(upper - value)

    def barrier_diff(self, value, lower, upper):
        """
        Jacobian:
        b * -1 / (a - x)^2 upper limit
        b * 1/ (a - x)^2 lower limit
        """
        return -1 / ((lower - value)**2), 1 / ((upper - value)**2)

    def barrier_hess(self, value, lower, upper):
        """
        Hessian:
        b * -2 / (a - x)^3 upper limit
        b * 2 / (a - x)^3 lower limit
        """
        return -2 / ((lower - value)**3), 2 / ((upper - value)**3)

class HalfQuadraticBarrierCost(BarrierCost):
    @property
    def is_quad(self) -> bool:
        return False

    @property
    def is_convex(self) -> bool:
        return True

    @property
    def is_diff(self) -> bool:
        return True

    @property
    def is_twice_diff(self) -> bool:
        return True
    @ property
    def has_goal(self) -> bool:
        return False

    @property
    def is_hard(self) -> bool:
        return False

    def barrier(self, value, lower, upper):
        """
        Cost Function:
        b = scale
        b * ( -a + x )**2 upper limit x < a
        b * (  a - x )**2 lower limit x > a := -x < -a"""
        return (-lower + value)**2,  (upper - value)**2

    def barrier_diff(self, value, lower, upper):
        """
        Jacobian:
        b * 2 * (-a + x) upper limit
        b * 2 * ( a - x) lower limit
        """
        return 2* (-lower + value),  2*(-upper + value)

    def barrier_hess(self, value, lower, upper):
        """
        Hessian:
        b * 2  upper limit
        b * 2   lower limit
        """
        return 2, 2