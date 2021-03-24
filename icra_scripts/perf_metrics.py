# Created by William Edwards (wre2@illinois.edu), 2021-01-09

# Standard library includes

# External library includes
import numpy as np
import numpy.linalg as la

def threshold_metric(goal, obs_range, threshold, traj):
    cost = 0.0
    for i in range(len(traj)):
        if la.norm(traj[i].obs[obs_range[0]:obs_range[1]] - goal, 1) > threshold:
            cost += 1
    return cost
