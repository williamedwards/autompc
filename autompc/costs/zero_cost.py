# Created by Kris Hauser, (kkhauser@illinois.edu)

import numpy as np
from .cost import Cost

class ZeroCost(Cost):
    """
    A zero cost.
    """
    def __init__(self, system, **properties):
        Cost.__init__(self,system,quad=True,convex=True,diff=True,twice_diff=True)

    def __call__(self, traj):
        return 0.0

    def incremental(self, obs, control, t=None) -> float:
        return 0.0

    def incremental_diff(self, obs, ctrl, t=None):
        return 0.0,np.zeros(len(obs)),np.zeros(len(ctrl))

    def incremental_hess(self, obs, ctrl, t=None):
        return 0.0,np.zeros(len(obs)),np.zeros(len(ctrl)),np.zeros((len(obs),len(obs))),np.zeros((len(obs),len(ctrl))),np.zeros((len(ctrl),len(ctrl)))

    def terminal(self, obs, t=None) -> float:
        return 0.0

    def terminal_diff(self, obs, t=None):
        return 0.0,np.zeros(len(obs))

    def terminal_hess(self, obs, t=None):
        return 0.0,np.zeros(len(obs)),np.zeros((len(obs),len(obs)))

