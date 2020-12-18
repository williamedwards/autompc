import numpy as np


def normalize(means, std, A):
    At = []
    for i in range(A.shape[1]):
        At.append((A[:,i] - means[i]) / std[i])
    return np.vstack(At).T

def get_normalized_model_rmse(model, trajs, horiz=1):
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

def get_variance_score(models, trajs):
    #baseline_std = np.zeros((tinf.system.obs_dim))
    deltass = []
    for traj in trajs:
        deltas = traj.obs[1:, :] - traj.obs[:-1, :]
        deltass.append(deltas)
    deltas = np.concatenate(deltass)
    baseline_std = np.std(deltas, axis=0)

    pred_deltass = []
    for traj in trajs:
        pred_deltas = np.zeros((len(traj)-1, traj.system.obs_dim, len(models)))
        for i, model in enumerate(models):
            preds = model.pred_parallel(traj.obs[:-1, :], traj.ctrls[:-1, :])
            pred_deltas[:, :, i] = preds - traj.obs[:-1, :]
        pred_deltass.append(pred_deltas)
    pred_deltas = np.concatenate(pred_deltass)
    pred_std = np.std(pred_deltas, axis=2)
    pred_std_mean = np.mean(pred_std, axis=0)

    score = np.mean(pred_std_mean / baseline_std)
    print("Score: ", score)
    return score

def get_model_variance(models, trajs):
    dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
    dy_means = np.mean(dY, axis=0)
    dy_std = np.std(dY, axis=0)

    pred_deltass = []
    for traj in trajs:
        pred_deltas = np.zeros((len(traj)-1, traj.system.obs_dim, len(models)))
        for i, model in enumerate(models):
            pred = model.pred_parallel(traj.obs[:-1, :], traj.ctrls[:-1, :])
            pred_delta = pred - traj.obs[:-1, :]
            pred_delta_norm = normalize(dy_means, dy_std, pred_delta)
            pred_deltas[:, :, i] = pred_delta_norm
        pred_deltass.append(pred_deltas)
    pred_deltas = np.concatenate(pred_deltass)
    pred_std = np.std(pred_deltas, axis=2)
    pred_std_mean = np.mean(pred_std, axis=None)

    return pred_std_mean
