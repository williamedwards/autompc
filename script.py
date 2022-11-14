from distutils.log import info
import os, glob
import pickle
import json
from collections import namedtuple
import numpy as np

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

from autompc.benchmarks import CartpoleSwingupV2Benchmark, benchmark
from autompc import AutoSelectController
from autompc.tuning import ControlTuner
from autompc.sysid import MLP

from autompc.model_metalearning.meta_utils import load_data

import xml.etree.ElementTree as ET
import tempfile

path = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/temp.json'

T = []
with open(path, 'r') as f:
    data = json.load(f)
    # print(data['foo'])

names = []
configs = []
for d in data['foo']:
    name = d['env']
    config = d['final_config']
    names.append(name)
    configs.append(config)

data_path = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/meta_data'
system, trajs = load_data(data_path, names[0])
# config = CS(configs[1])
print(type(config))
# print(config)

