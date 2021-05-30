import numpy as np
from .. import zeros
from pdb import set_trace


def normalize(means, std, A):
    At = []
    for i in range(A.shape[1]):
        At.append((A[:,i] - means[i]) / std[i])
    return np.vstack(At).T

def get_model_rmse(model, trajs, horizon=1):
    """
    Computes (unnormalized) RMSE at fixed horizon

    Parameters
    ----------
    model : Model
        Model class to consider

    trajs : List of Trajectories
        Trajectories on which to evaluate

    horizon : int
        Prediction horizon at which to evaluate.
        Default is 1.
    """
    sqerrss = []
    for traj in trajs:
        if hasattr(model, "traj_to_states"):
            state = model.traj_to_states(traj[:-horizon])
        else:
            state = traj.obs[:-horizon, :]
        for k in range(horizon):
            state = model.pred_batch(state, traj.ctrls[k:-(horizon-k), :])
        if hasattr(model, "traj_to_states"):
            state = state[:,:model.system.obs_dim]
        actual = traj.obs[horizon:]
        sqerrs = (state - actual) ** 2
        sqerrss.append(sqerrs)
    sqerrs = np.concatenate(sqerrss)
    rmse = np.sqrt(np.mean(sqerrs, axis=None)*trajs[0].system.obs_dim)
    return rmse

def get_model_rmsmens(model, trajs, horiz=1):
    """
    Compute root mean squared model error, normalized step-wise (RMSMENS).

    TODO equation.
    """
    dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
    dy_means = np.mean(dY, axis=0)
    dy_std = np.std(dY, axis=0)

    sqerrss = []
    for traj in trajs:
        state = traj.obs[:-horiz, :]
        for k in range(horiz):
            pstate = state
            state = model.pred_parallel(state, traj.ctrls[k:-(horiz-k), :])
        pred_deltas = state - pstate
        act_deltas = traj.obs[horiz:] - traj.obs[horiz-1:-1]
        norm_pred_deltas = normalize(dy_means, dy_std, pred_deltas)
        norm_act_deltas = normalize(dy_means, dy_std, act_deltas)
        sqerrs = (norm_pred_deltas - norm_act_deltas) ** 2
        sqerrss.append(sqerrs)
    sqerrs = np.concatenate(sqerrss)
    rmse = np.sqrt(np.mean(sqerrs, axis=None))
    return rmse
