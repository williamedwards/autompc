# Created by William Edwards (wre@illinois.edu), 2020-10-21

# Standard library includes
import os, sys
from pdb import set_trace
import argparse

# External projects include
import numpy as np
import gym
#import gym_cartpole_swingup
import custom_gym_cartpole_swingup
from custom_gym_cartpole_swingup.envs.cartpole_swingup import State as GymCartpoleState

# Internal project includes
import autompc as ampc
from autompc.tasks import Task, QuadCost

from collections import namedtuple

TaskInfo = namedtuple("TaskInfo", ["system", "task", "init_obs", "env_name", 
    "set_env_state", "gym_to_obs", "gym_to_ctrl", "ctrl_to_gym"])

def cartpole_swingup_task():
    system = ampc.System(["theta", "omega", "x", "dx"], ["u"])
    system.dt = 0.01
    Q = np.eye(4)
    R = 0.01 * np.eye(1)
    F = 100.0 * np.eye(4)
    cost = QuadCost(system, Q, R, F)
    task = Task(system)
    task.set_cost(cost)
    task.set_ctrl_bound("u", -1.0, 1.0)
    init_obs = np.array([3.1, 0.0, 0.0, 0.0])
    env_name = "CustomCartPoleSwingUp-v1"
    def gym_to_obs(gym):
        x_pos, x_dot, cos, sin, omega = gym
        theta = np.arctan2(sin, cos)
        return np.array([theta, omega, x_pos, x_dot])
    def gym_to_ctrl(action):
        return action[:]
    def ctrl_to_gym(ctrl):
        return ctrl[:]
    def set_env_state(env, obs):
        theta, thetadot, x, xdot = obs
        env.env.state = GymCartpoleState(x, xdot, theta, thetadot)
    return TaskInfo(system=system, 
            task=task, 
            init_obs=init_obs, 
            env_name=env_name, 
            set_env_state=set_env_state,
            gym_to_obs=gym_to_obs, 
            gym_to_ctrl=gym_to_ctrl, 
            ctrl_to_gym=ctrl_to_gym)

def init_task(task):
    if task == "cartpole-swingup":
        return cartpole_swingup_task()
    else:
        raise
