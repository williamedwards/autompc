from distutils.log import info
import os, glob
import pickle
from collections import namedtuple
import numpy as np

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

import xml.etree.ElementTree as ET
import tempfile

save_path = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/gym_size_ext_xml/'
model_path=os.path.dirname(gym.envs.__file__) + "/mujoco/assets/half_cheetah.xml"
tree = ET.parse(model_path)
print(tree)
body_part = ['torso']
size_scale = 1.5

geom = tree.find(".//geom[@name='%s']" % body_part[0])
print(geom)

sizes  = [float(x) for x in geom.attrib["size"].split(" ")]
print(geom.attrib["size"])

geom.attrib["size"] = " ".join([str(x * size_scale) for x in sizes ])
print(geom.attrib["size"])

file_name = 'half_cheetah_' + 'torso_' + str(size_scale) + ".xml"
output_file = save_path + file_name
tree.write(output_file)

class NewEnv(HalfCheetahEnv, utils.EzPickle):
    
    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, output_file, 5, observation_space=observation_space, **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

gym.envs.register(
    id = 'HalfCheetahBigTorso-v2',
    entry_point="script:NewEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

env = gym.make('HalfCheetahBigTorso-v2')