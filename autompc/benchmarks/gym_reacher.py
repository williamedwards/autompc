

import numpy as np

from ..system import System
from .gym_mujoco import GymMujocoBenchmark, GymRewardCost, gym_dynamics, _get_init_obs


def reacher_augment(env, x):
    finger_x, finger_y, finger_z = env.get_body_com("fingertip")
    x_aug = np.concatenate([x, [finger_x, finger_y]])
    return x_aug

def gym_reacher_dynamics(env, x, u, n_frames):
    """ Augment observation with end-ee position """
    x_noaug = x[:-2]
    new_x_noaug = gym_dynamics(env, x_noaug, u, n_frames)
    new_x = reacher_augment(env, new_x_noaug)
    return new_x

def _get_reacher_init_obs(env):
    init_obs_noaug = _get_init_obs(env)
    init_obs = reacher_augment(env, init_obs_noaug)
    return init_obs

class GymReacherCost(GymRewardCost):
    def __init__(self, system, env):
        super().__init__(system, env)
        goal = np.zeros(system.obs_dim)
        goal[-2:] = env.get_body_com("target")[:2]
        self.properties["goal"] = goal


class GymReacherBenchmark(GymMujocoBenchmark):
    def __init__(self, data_gen_method="uniform_random"):
        super().__init__(name="Reacher-v2", data_gen_method=data_gen_method)

    def _get_init_obs(self):
        return _get_reacher_init_obs(self.env)

    def _get_system(self):
        system_noaug = super()._get_system()
        system = System(
            observations = system_noaug.observations + ["finger_x", "finger_y"],
            controls = system_noaug.controls,
            dt = system_noaug.dt
        )
        return system

    def _get_cost(self, system):
        return GymReacherCost(system, self.env)

    def dynamics(self, x, u):
        return gym_reacher_dynamics(self.env,x,u,n_frames=self.env.frame_skip)