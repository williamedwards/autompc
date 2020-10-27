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
from acrobot_task import acrobot_swingup_task
from pipelines import init_pipeline
from sysid1 import runexp_sysid1
from sysid2 import runexp_sysid2
from tuning1 import runexp_tuning1
from cost_tuning import runexp_cost_tuning
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
    elif task_name == "acrobot-swingup":
        return acrobot_swingup_task()
    else:
        raise ValueError("Task not found")


def main(args):
    if args.command == "sysid1":
        Model = init_model(args.model)
        tinf = init_task(args.task)
        result = runexp_sysid1(Model, tinf, tune_iters=args.tuneiters,
                seed=args.seed)
        save_result(result, "sysid1", args.task, args.model, 
                args.tuneiters, args.seed)
    elif args.command == "tuning1":
        tinf = init_task(args.task)
        pipeline = init_pipeline(tinf, args.pipeline)
        result = runexp_tuning1(pipeline, tinf, tune_iters=args.tuneiters,
                seed=args.seed)
        save_result(result, "tuning1", args.task, args.pipeline,
                args.tuneiters, args.seed)
    elif args.command == "sysid2":
        tinf = init_task(args.task)
        pipeline = init_pipeline(tinf, args.pipeline)
        result = runexp_sysid2(pipeline, tinf, tune_iters=args.tuneiters,
                sub_exp = args.subexp, seed=args.seed)
        save_result(result, "sysid2", args.task, args.pipeline, args.subexp,
                args.tuneiters, args.seed)
    elif args.command == "cost_tuning":
        tinf = init_task(args.task)
        pipeline = init_pipeline(tinf, args.pipeline)
        result = runexp_cost_tuning(pipeline, tinf, tune_iters=args.tuneiters,
                seed=args.seed)
        save_result(result, "cost_tuning", args.task, args.pipeline,
                args.tuneiters, args.seed)
    else:
        raise ValueError("Command not recognized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str)
    parser.add_argument("--task", default="cartpole-swingup")
    parser.add_argument("--pipeline", default="")
    parser.add_argument("--tuneiters", default=5, type=int)
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--subexp", default=1, type=int)
    args = parser.parse_args()
    main(args)
