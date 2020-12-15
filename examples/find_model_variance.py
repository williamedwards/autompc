import time
from pdb import set_trace
import ConfigSpace as CS
import sys, os, io
sys.path.append(os.getcwd() + "/..")
sys.path.append(os.getcwd() + "/../icra_scripts")

import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.linalg as la
import argparse
import pickle
from joblib import Memory
import torch

from scipy.integrate import solve_ivp
from cartpole_control_test import animate_cartpole
from test_shared import animate_pendulum
from tuning1 import runsim
from pendulum_task import pendulum_swingup_task
from cartpole_task import cartpole_swingup_task
from halfcheetah_task import halfcheetah_task, env
from pipelines import init_mlp_ilqr, init_halfcheetah
from autompc.sysid import MLP

memory = Memory("cache")

def get_halfch_cfgs(tinf):
    cs = MLP.get_configuration_space(tinf.system)
    cfg1 = cs.get_default_configuration()
    cfg1["n_hidden_layers"] = "3"
    cfg1["hidden_size_1"] = 128
    cfg1["hidden_size_2"] = 128
    cfg1["hidden_size_3"] = 128

    cfg2 = cs.get_default_configuration()
    cfg2["n_hidden_layers"] = "3"
    cfg2["hidden_size_1"] = 64
    cfg2["hidden_size_2"] = 64
    cfg2["hidden_size_3"] = 64

    cfg3 = cs.get_default_configuration()
    cfg3["n_hidden_layers"] = "2"
    cfg3["hidden_size_1"] = 64
    cfg3["hidden_size_2"] = 64

    cfg4 = cs.get_default_configuration()
    cfg4["n_hidden_layers"] = "2"
    cfg4["hidden_size_1"] = 32
    cfg4["hidden_size_2"] = 32

    return [cfg1, cfg2, cfg3, cfg3]

@memory.cache
def train_mlp_inner(system, trajs, cfg_vals):
    cs = MLP.get_configuration_space(system)
    cfg = CS.Configuration(cs, cfg_vals)
    model = ampc.make_model(system, MLP, cfg, use_cuda=True)
    model.train(trajs)
    return model.get_parameters()

def train_mlp(system, trajs, cfg_vals):
    cs = MLP.get_configuration_space(system)
    cfg = CS.Configuration(cs, cfg_vals)
    model = ampc.make_model(system, MLP, cfg, use_cuda=True)
    params = train_mlp_inner(system, trajs, cfg_vals)
    model.set_parameters(params)
    model.net = model.net.to("cpu")
    model._device = "cpu"
    return model

def get_variance_score(tinf, models, trajs):
    #baseline_std = np.zeros((tinf.system.obs_dim))
    deltass = []
    for traj in trajs:
        deltas = traj.obs[1:, :] - traj.obs[:-1, :]
        deltass.append(deltas)
    deltas = np.concatenate(deltass)
    baseline_std = np.std(deltas, axis=0)

    pred_deltass = []
    for traj in trajs:
        pred_deltas = np.zeros((len(traj)-1, tinf.system.obs_dim, len(models)))
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

def get_model_rmse(model, trajs):
    sqerrss = []
    for traj in trajs:
        preds = model.pred_parallel(traj.obs[:-1, :], traj.ctrls[:-1, :])
        sqerrs = (preds - traj.obs[1:]) ** 2
        sqerrss.append(sqerrs)
    np.concatenate(sqerrss)
    rmse = np.sqrt(np.mean(sqerrs, axis=None))
    return rmse

@memory.cache
def evaluate_halfch_cfg(cfg_vals, seed, n_models=10):
    tinf = halfcheetah_task()
    rng = np.random.default_rng(seed)
    holdout_seed = rng.integers(1 << 30)
    model_seeds = [rng.integers(1 << 30) for _ in range(n_models)]
    holdout_set = tinf.gen_surr_trajs(seed)
    trajss = [tinf.gen_surr_trajs(seed) for seed in model_seeds]
    models = [train_mlp(tinf.system, trajs, cfg_vals) for trajs in trajss]
    variance_score = get_variance_score(tinf, models, holdout_set)
    model_accuracy = get_model_rmse(models[0], holdout_set)
    return variance_score, model_accuracy

def evaluate_random_cfgs(tinf, seed, n_cfgs):
    rng = np.random.default_rng(seed)
    traj_gen_seed = rng.integers(1 << 30)
    cs = MLP.get_configuration_space(tinf.system)
    cs.seed(rng.integers(1 << 30))
    cfgs = []
    scores = []
    for _ in range(n_cfgs):
        cfg = cs.sample_configuration()
        cfg_vals = cfg.get_dictionary()
        score = evaluate_halfch_cfg(cfg_vals, traj_gen_seed, n_models=10)
        cfgs.append(cfg)
        scores.append(score)

    for cfg, (var_score, model_accuracy) in zip(cfgs, scores):
        print("Cfg")
        print(cfg)
        print("Variance Score: ", var_score)
        print("Model Accuracy", model_accuracy)
        print("===============")

    var_scores = [var_score for var_score, model_accuracy in scores]
    model_accuracies = [model_accuracy for var_score, model_accuracy in scores]
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("Model Accuracy vs Variance")
    ax.scatter(model_accuracies, var_scores)
    ax.scatter([0.605], [0.0663], color="r")
    ax.set_xlabel("Model Accuracy")
    ax.set_ylabel("Variance Score")

    plt.show()

    set_trace()

def main(sysname, seed, n_models=10):
    rng = np.random.default_rng(seed)
    holdout_seed = rng.integers(1 << 30)
    model_seeds = [rng.integers(1 << 30) for _ in range(n_models)]
    if sysname == "cartpole":
        tinf = cartpole_swingup_task()
    elif sysname == "halfcheetah":
        tinf = halfcheetah_task()
    elif sysname == "pendulum":
        tinf = pendulum_swingup_task()
    holdout_set = tinf.gen_surr_trajs(seed)
    trajss = [tinf.gen_surr_trajs(seed) for seed in model_seeds]
    models = [train_mlp(tinf.system, trajs) for trajs in trajss]
    get_variance_score(tinf, models, holdout_set)

def main2():
    tinf = halfcheetah_task()
    if len(sys.argv) == 1:
        evaluate_random_cfgs(tinf, seed=101, n_cfgs=15)
    else:
        cfg = get_halfch_cfgs(halfcheetah_task())[int(sys.argv[1])]
        print(evaluate_halfch_cfg(cfg.get_dictionary(), 220, n_models=10))

if __name__ == "__main__":
    #main("pendulum", 80)
    main2()

