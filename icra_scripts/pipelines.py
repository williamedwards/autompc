# Standard library includes
import os, sys
from pdb import set_trace
import argparse

# External projects include
import numpy as np
import gym
import gym_cartpole_swingup
from gym_cartpole_swingup.envs.cartpole_swingup import State as GymCartpoleState

# Internal project includes
import autompc as ampc
from autompc.tasks.quad_cost_transformer import QuadCostTransformer
from autompc.tasks.half_cheetah_transformer import HalfCheetahTransformer
from autompc.tasks.swimmer_transformer import SwimmerTransformer
from autompc.pipelines import FixedControlPipeline
from autompc.sysid import Koopman, MLP
from autompc.control import FiniteHorizonLQR, IterativeLQR

def init_mlp_ilqr(tinf):
    pipeline = FixedControlPipeline(tinf.system, tinf.task, MLP, 
            IterativeLQR, [QuadCostTransformer],
            controller_kwargs={"reuse_feedback" : -1})
    return pipeline

def init_halfcheetah(tinf):
    pipeline = FixedControlPipeline(tinf.system, tinf.task, MLP, 
            IterativeLQR, [HalfCheetahTransformer],
            controller_kwargs={"reuse_feedback" : -1},
            use_cuda=True)
    return pipeline

def init_swimmer(tinf):
    pipeline = FixedControlPipeline(tinf.system, tinf.task, MLP, 
            IterativeLQR, [SwimmerTransformer],
            controller_kwargs={"reuse_feedback" : -1},
            use_cuda=True)
    return pipeline

def init_pipeline(tinf, pipeline):
    if pipeline == "mlp-ilqr":
        return init_mlp_ilqr(tinf)
    elif pipeline == "halfcheetah":
        return init_halfcheetah(tinf)
    elif pipeline == "swimmer":
        return init_swimmer(tinf)
    else:
        raise ValueError("Unrecognized Pipeline")
