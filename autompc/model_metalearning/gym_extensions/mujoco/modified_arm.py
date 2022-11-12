import numpy as np
from gym import utils
from gym.spaces import Box
from gym.envs.mujoco import MuJocoPyEnv

from gym.envs.mujoco.pusher import PusherEnv

import os
import gym
import random
import six

class PusherMovingGoalEnv(PusherEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """
    def __init__(self, model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/pusher.xml", **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, model_path, 5, observation_space=observation_space
        )
        utils.EzPickle.__init__(self)

        # make sure we're using a proper OpenAI gym Mujoco Env
        assert isinstance(self, MuJocoPyEnv)

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.concatenate([
                self.np_random.uniform(low=-0.2, high=0, size=1),
                self.np_random.uniform(low=-0.1, high=0.3, size=1)])

        while True:
            self.cylinder_pos = np.concatenate([
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1)])
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005,
                high=0.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()



