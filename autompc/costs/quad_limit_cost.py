# Created by Teodor Tchalakov, (ttcha2@illinois.edu)

import numpy as np

from .cost import Cost

class QuadLimitCost(Cost):
    def __init__(self, system, boundedStates, Q, R, F=None, goal=None):
        """
        Create quadratic cost.

        Parameters
        ----------
        system : System
            System for cost

        Q : numpy array of shape (self.obs_dim, self.obs_dim)
            Observation cost matrix

        R : numpy array of shape (self.ctrl_dim, self.ctrl_dim)
            Control cost matrix

        F : numpy array of shape (self.ctrl_dim, self.ctrl_dim)
            Terminal observation cost matrix

        goal : numpy array of shape self.obs_dim
            Goal state. Default is zero state
        """
        super().__init__(system)
        if Q.shape != (system.obs_dim, system.obs_dim):
            raise ValueError("Q is the wrong shape")
        if R.shape != (system.ctrl_dim, system.ctrl_dim):
            raise ValueError("R is the wrong shape")
        if not F is None:
            if F.shape != (system.obs_dim, system.obs_dim):
                raise ValueError("F is the wrong shape")
        else:
            F = np.zeros((system.obs_dim, system.obs_dim))

        self._Q = np.copy(Q)
        self._R = np.copy(R)
        self._F = np.copy(F)
        if goal is None:
            goal = np.zeros(system.obs_dim)
        self._goal = np.copy(goal)

        #Barrier Cost Setup
        self.obsConfiguration = []
        self.ctrlsConfiguration = []

        for variable in boundedStates.keys():
            if(variable in system.observations):
                self.obsConfiguration.append([variable, boundedStates[variable]])
            if(variable in system.controls):
                self.ctrlsConfiguration.append([variable, boundedStates[variable]])
        
        self._is_quad = False
        self._is_convex = False
        self._is_diff = True
        self._is_twice_diff = True
        self._has_goal = True

    #Cost Function:
    # - b * ln ( a - x ) upper limit
    # - b * ln ( a + x ) lower limit
    def eval_obs_cost(self, obs):
        sum = 0
        for boundedObs in self.obsConfiguration:
            index = self.system.observations.index(boundedObs[0])
            limit, scale, upper = boundedObs[1]
            self._direction = -1
            if(upper):
                direction = 1
            sum = sum + -scale * np.log(limit - (direction * obs[index]))

        obst = obs - self._goal
        quadCost = obst.T @ self._Q @ obst
        return sum + quadCost

    #Jacobian:
    # b / (a - x) upper limit
    # -b / (a - x) lower limit
    def eval_obs_cost_diff(self, obs):
        jacobian = np.zeros(self.system.obs_dim)
        for boundedObs in self.obsConfiguration:
            index = self.system.observations.index(boundedObs[0])
            limit, scale, upper = boundedObs[1]
            self._direction = -1
            if(upper):
                direction = 1
            jacobian[index] = direction * scale / (limit - obs[index])

        obst = obs - self._goal
        quadCost = (self._Q + self._Q.T) @ obst
        return jacobian + quadCost

    #Hessian:
    # b / (a - x)^2 upper limit
    # b / (a - x)^2 lower limit
    def eval_obs_cost_hess(self, obs):
        hessian = np.zeros((self.system.obs_dim, self.system.obs_dim))
        for boundedObs in self.obsConfiguration:
            index = self.system.observations.index(boundedObs[0])
            limit, scale, upper = boundedObs[1]
            hessian[index][index] = scale / ((limit - obs[index])**2)
        obst = obs - self._goal
        quadCost = self._Q + self._Q.T
        return hessian + quadCost

    def eval_ctrl_cost(self, ctrl):
        sum = 0
        for boundedCtrl in self.ctrlsConfiguration:
            index = self.system.controls.index(boundedCtrl[0])
            limit, scale, upper = boundedCtrl[1]
            self._direction = -1
            if(upper):
                direction = 1
            sum = sum + -scale * np.log(limit - (direction * ctrl[index]))
        quadCost = ctrl.T @ self._R @ ctrl
        return sum + quadCost
    
    def eval_ctrl_cost_diff(self, ctrl):
        jacobian = np.zeros(self.system.ctrl_dim)
        for boundedCtrl in self.ctrlsConfiguration:
            index = self.system.controls.index(boundedCtrl[0])
            limit, scale, upper = boundedCtrl[1]
            self._direction = -1
            if(upper):
                direction = 1
            jacobian[index] = direction * scale / (limit - ctrl[index])
        quadCost = (self._R + self._R.T) @ ctrl
        return jacobian + quadCost

    def eval_ctrl_cost_hess(self, ctrl):
        hessian = np.zeros((self.system.ctrl_dim, self.system.ctrl_dim))
        for boundedCtrl in self.ctrlsConfiguration:
            index = self.system.controls.index(boundedCtrl[0])
            limit, scale, upper = boundedCtrl[1]
            hessian[index][index] = scale / ((limit - ctrl[index])**2)
        quadCost = self._R + self._R.T
        return hessian + quadCost

    def eval_term_obs_cost(self, obs):
        obst = obs - self._goal
        return obst.T @ self._F @ obst

    def eval_term_obs_cost_diff(self, obs):
        obst = obs - self._goal
        return (self._F + self._F.T) @ obs

    def eval_term_obs_cost_hess(self, obs):
        return self._F + self._F.T