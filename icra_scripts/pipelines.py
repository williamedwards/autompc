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
from autompc.pipelines import FixedControlPipeline
from autompc.sysid import Koopman, MLP
from autompc.control import FiniteHorizonLQR, IterativeLQR

def init_mlp_ilqr(tinf)
    pipeline = FixedControlPipeline(tinf.system, tinf.task, MLP, 
            IterativeLQR, [QuadCostTransformer])
    return pipeline

def init_pipeline(tinf, pipeline):
    if pipeline == "mlp-ilqr":
        return init_mlp_ilqr(tinf)
