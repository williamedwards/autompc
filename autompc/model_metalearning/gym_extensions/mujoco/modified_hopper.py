import numpy as np
import os

import gym
from gym import utils
from gym.spaces import Box
from gym.envs.mujoco import MuJocoPyEnv
from gym.envs.mujoco.hopper import HopperEnv

from autompc.model_metalearning.gym_extensions.mujoco.wall_envs import *
from autompc.model_metalearning.gym_extensions.mujoco.perturbed_bodypart_env import ModifiedSizeEnvFactory
from autompc.model_metalearning.gym_extensions.mujoco.gravity_envs import GravityEnvFactory

HopperSimpleWallEnv = lambda *args, **kwargs : SimpleWallEnvFactory(ModifiedHopperEnv)(model_path=os.path.dirname(gym.envs.__file__) + "/mujoco/assets/hopper.xml", ori_ind=-1, *args, **kwargs)

# HopperStairs = lambda *args, **kwargs : StairsFactory(ModifiedHopperEnv)(model_path=os.path.dirname(gym.envs.__file__) + "/mujoco/assets/hopper.xml", ori_ind=-1, *args, **kwargs)

HopperGravityEnv = lambda *args, **kwargs : GravityEnvFactory(ModifiedHopperEnv)(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/hopper.xml", *args, **kwargs)

HopperModifiedBodyPartSizeEnv = lambda *args, **kwargs : ModifiedSizeEnvFactory(ModifiedHopperEnv)(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/hopper.xml", *args, **kwargs)


class ModifiedHopperEnv(HopperEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """
    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, kwargs["model_path"], 4, observation_space=observation_space
        )
        utils.EzPickle.__init__(self)

class HopperWithSensorEnv(HopperEnv, utils.EzPickle):
    """
    Adds empty sensor readouts, this is to be used when transfering to WallEnvs where we get sensor readouts with distances to the wall
    """

    def __init__(self, n_bins=10, **kwargs):
        self.n_bins = n_bins
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, kwargs["model_path"], 4, observation_space=observation_space
        )
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        obs = np.concatenate([
            HopperEnv._get_obs(self),
            np.zeros(self.n_bins)
            # goal_readings
        ])
        return obs
