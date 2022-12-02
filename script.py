from distutils.log import info
import os, glob
import pickle
import json
from collections import namedtuple
import numpy as np
import pandas as pd

import ConfigSpace as CS

import gym
import mujoco_py
from gym.envs.mujoco import MuJocoPyEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym import utils
from gym.spaces import Box


from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger

from autompc import AutoSelectController
from autompc.tuning import ControlTuner
from autompc.sysid import MLP
from autompc.sysid.metrics import get_model_rmse,get_model_rmsmens
from autompc.sysid.autoselect import AutoSelectModel
from autompc.tuning.model_evaluator import CrossValidationModelEvaluator, HoldoutModelEvaluator, ModelEvaluator

from autompc.model_metalearning.meta_utils import load_data, load_cfg

import xml.etree.ElementTree as ET
import tempfile

# name = "HalfCheetah-v2"
data_path = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/meta_data'
cfg_path = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/meta_cfg'
# system, trajs = load_data(data_path, name)
# cfg = load_cfg(cfg_path, name)
# print(cfg)

# model = AutoSelectModel(system)
# model.set_config(cfg)

eval_metric="rmse"
eval_horizon=1
eval_quantile=None
eval_folds=3
# evaluator = CrossValidationModelEvaluator(trajs, eval_metric, horizon=eval_horizon, quantile=eval_quantile, num_folds=eval_folds,
#                     rng=np.random.default_rng(100))
# score = evaluator(model)
# print(score)

names = ["HalfCheetah-v2", "HalfCheetahSmall-v2"]
output_results_dictionary = {}

# data
for data_name in names:
    print(data_name)
    system, trajs = load_data(data_path, data_name)
    scores = []
    # config
    for cfg_name in names:
        cfg = load_cfg(cfg_path, cfg_name)
        print(cfg)
        model = AutoSelectModel(system)
        model.set_config(cfg)
        evaluator = CrossValidationModelEvaluator(trajs, eval_metric, horizon=eval_horizon, quantile=eval_quantile, num_folds=eval_folds,
                    rng=np.random.default_rng(100))
        score = evaluator(model)
        scores.append(score)
    output_results_dictionary[data_name] = scores
        
print(output_results_dictionary)
        

