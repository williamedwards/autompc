# Created by William Edwards (wre2@illinois.edu), 2020-10-26

# Standard library includes
import os, sys
import argparse
import pickle
from pdb import set_trace

# External project includes
import numpy as np

# Internal project includes
import autompc as ampc
from autompc.sysid import MLP, ARX, Koopman, SINDy, ApproximateGaussianProcess

from cartpole_task import cartpole_swingup_task
from pendulum_task import pendulum_swingup_task
from sysid1 import runexp_sysid1
from utils import *



def init_model(model_name):
    if model_name == "mlp":
        return MLP
    elif model_name == "arx":
        return ARX
    elif model_name == "koop":
        return Koopman
    elif model_name == "sindy":
        return SINDy
    elif model_name == "approxgp":
        return ApproximateGaussianProcess
    else:
        raise ValueError("Model not found")

def init_task(task_name):
    if task_name == "cartpole-swingup":
        return cartpole_swingup_task()
    elif task_name == "pendulum-swingup":
        return pendulum_swingup_task()
    else:
        raise ValueError("Task not found")


def main(args):
    if args.command == "sysid1":
        Model = init_model(args.model)
        tinf = init_task(args.task)
        results = runexp_sysid1(Model, tinf, tune_iters=args.tuneiters,
                seed=args.seed)
        save_result(results, "sysid1", args.task, args.model, 
                args.tuneiters, args.seed)
    else:
        raise ValueError("Command not recognized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("--task", default="cartpole-swingup")
    parser.add_argument("--tuneiters", default=5, type=int)
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    main(args)
