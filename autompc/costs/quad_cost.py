# Created by William Edwards, (wre2@illinois.edu)

import numpy as np

from .cost import Cost

class QuadCost(Cost):
    def __init__(self, system, Q, R, F=None, goal=None):
        """
        Create quadratic cost.  Cost is:
        
            \sum_i (x[i]-xg)^T Q (x[i]-xg) + u[i]^T R u[i] + (x[T]-xg)^T F (x[T]-xg)
        
        where xg is a goal state (may be None, in which case it is treated
        as zero).

        Parameters
        ----------
        system : System
            System for cost

        Q : numpy array of shape (self.obs_dim, self.obs_dim)
            Observation cost matrix

        R : numpy array of shape (self.ctrl_dim, self.ctrl_dim)
            Control cost matrix

        F : numpy array of shape (self.obs_dim, self.obs_dim)
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
        self.properties['goal'] = np.copy(goal)
        
        self.properties['quad'] = True
        self.properties['convex'] = True
        self.properties['diff'] = True
        self.properties['twice_diff'] = True

    def get_cost_matrices(self):
        """
        Return quadratic Q, R, and F matrices.
        """
        return np.copy(self._Q), np.copy(self._R), np.copy(self._F)
    
    def incremental(self, obs, control):
        try:
            obst = obs - self.goal
        except:
            obst = obs
        return obst.T @ self._Q @ obst + control.T @ self._R @control

    def incremental_diff(self, obs, control):
        try:
            obst = obs - self.goal
        except:
            obst = obs
        return obst.T @ self._Q @ obst + control.T @ self._R @control, (self._Q + self._Q.T) @ obst, (self._R + self._R) @ control
    
    def incremental_hess(self, obs, control):
        try:
            obst = obs - self.goal
        except:
            obst = obs
        QQt = (self._Q + self._Q.T)
        RRt = (self._R + self._R)
        hess_obs_ctrl = np.zeros((self.system.obs_dim, self.system.ctrl_dim))
        return obst.T @ self._Q @ obst + control.T @ self._R @control, QQt @ obst, RRt @ control, QQt, hess_obs_ctrl, RRt
        
    def terminal(self, obs):
        try:
            obst = obs - self.goal
        except:
            obst = obs
        return obst.T @ self._F @ obst
    
    def terminal_diff(self, obs):
        try:
            obst = obs - self.goal
        except:
            obst = obs
        FFt = (self._F + self._F.T)
        return (obst.T @ self._F @ obst, FFt @ obst)
    
    def terminal_hess(self, obs):
        try:
            obst = obs - self.goal
        except:
            obst = obs
        FFt = (self._F + self._F.T)
        return (obst.T @ self._F @ obst, FFt @ obst, FFt)
    
    def __add__(self, rhs):
        if isinstance(rhs,QuadCost):
            if (self.goal is None and rhs.goal is None) or np.all(self.goal == rhs.goal):
                return QuadCost(self.system,self._Q+rhs._Q,self._R+rhs._R,self._F+rhs._F)
        return Cost.__add__(self,rhs)

    def __mul__(self, rhs):
        if not isinstance(rhs,(float,int)):
            raise ValueError("* only supports product with numbers")
        return QuadCost(self.system,self._Q*rhs,self._R*rhs,self._F*rhs,self.goal)
