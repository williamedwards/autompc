# Created by William Edwards (wre@illinois.edu), 2020-10-21

# Standard library includes
import os, sys
from pdb import set_trace
import argparse
import pickle

# External projects include
import numpy as np
import matplotlib.pyplot as plt

def load_tuning_results(results_dir, tuning_settings):
    results = []
    for label, task, surrogate, seed, tuneiters in tuning_settings:
        results_fn = f"{task}_{surrogate}_{seed}_{tuneiters}.pkl" 
        results_path = os.path.join(results_dir, results_fn)
        with open(results_path, "rb") as f:
            results.append((label, pickle.load(f)))
    return results

def make_plots(experiment_title, tuning_results, baselines):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(experiment_title + ": Surrogate Perf.")
    ax.set_xlabel("Tuning iterations")
    ax.set_ylabel("Surrogate Perfomance")
    n_iters = len(tuning_results[0][1]["inc_costs"])
    labels = []
    for label, res in tuning_results:
        perfs = [-cost for cost in res["inc_costs"]]
        ax.plot(perfs)
        labels.append(label)
    ax.legend(labels)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title(experiment_title + ": True Perf.")
    ax.set_xlabel("Tuning iterations")
    ax.set_ylabel("True Perf.")
    labels = []
    for label, value in baselines:
        ax.plot([0.0, n_iters], [value, value], "--")
        labels.append(label)
    for label, res in tuning_results:
        perfs = [-cost for cost in res["inc_truedyn_costs"]]
        ax.plot(perfs)
        labels.append(label)
    ax.legend(labels)

    plt.show()

def exp_cartpole1(results_dir):
    tuning_settings = [("True Dyn.", "cartpole-swingup", "truedyn", 42, 2)]
    baselines = [("BCQ", 500.0)]
    tuning_results = load_tuning_results(results_dir, tuning_settings)
    make_plots("Cartpole Swing-up", tuning_results, baselines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["cartpole1"])
    parser.add_argument("--resultsdir", default="results")
    args = parser.parse_args()
    
    if args.experiment == "cartpole1":
        exp_cartpole1(args.resultsdir)
    else:
        raise
