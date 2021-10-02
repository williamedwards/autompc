import numpy as np
from .. import zeros
from pdb import set_trace


def normalize(means, std, A):
    At = []
    for i in range(A.shape[1]):
        At.append((A[:,i] - means[i]) / std[i])
    return np.vstack(At).T

def get_model_residuals(model, trajs, horizon=1):
    resids = []
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
        resid = state - actual
        resids.append(resid)
    return np.vstack(resids)

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
    R"""
    Compute root mean squared model error, normalized step-wise (RMSMENS).

    Given a set of trajectories :math:`\mathcal{T}`, let :math:`x_{i,t} \in \mathbb{R}^n, u_{i,t} \in \mathbb{R}^m` denote the state and control input in the :math:`i^\textrm{th}` trajectory at time :math:`t`. 
    Let :math:`L_i` denote the length of the :math:`i\textrm{th}` trajectory.  Given a predictive model :math:`g`, let :math:`g(i,t,k)` give the prediction for :math:`x_{i,t+k}` given the 
    states :math:`\mathbf{x}_{i,1:t}` and controls :math:`\mathbf{u}_{i,1:t+k-1}`.
    Let

    .. math::
        \sigma = \textrm{std} \{ x_{i,t+1} - x_{i,t} \mid i=1,\ldots,\left|\mathcal{T}\right|
                        t=1,\ldots,L_i-1 \}
        \textrm{,}

    where the mean and standard deviation are computed element-wise.

    Denote the :math:`k`-step Root Mean Squared Model Error, Normalized Step-wise, RMSMENS(:math:`k`), of :math:`g` with respect to :math:`\mathcal{T}` as

    .. math::

      \sqrt{
       \frac{1}{n}
        \left\lVert
          \frac{1}{\left|\mathcal{T}\right|}
          \sum_{i=1}^{\left|\mathcal{T}\right|}
          \frac{1}{L_i-k  }
          \sum_{t=1}^{L_i-k}\frac{1}{\sigma^2} e(i,t,k)^2
        \right\rVert_1
      }
      \textrm{,}

    where

    .. math::
      e(i,t,k) = g(i,t,k) - g(i,t,k-1) - (x_{i,t+k} - x_{i,t+k-1})

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
