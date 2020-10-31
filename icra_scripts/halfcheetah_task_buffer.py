# Created by William Edwards (wre@illinois.edu), 2020-10-21

# Standard library includes
import os, sys
from pdb import set_trace
import argparse

# External projects include
import numpy as np
import numpy.linalg as la
import gym
import mujoco_py
from joblib import Memory
#import gym_cartpole_swingup
#import custom_gym_cartpole_swingup
#from custom_gym_cartpole_swingup.envs.cartpole_swingup import State as GymCartpoleState

# Internal project includes
import autompc as ampc
from autompc.tasks import Task, QuadCost

from collections import namedtuple

memory = Memory("cache")

TaskInfo = namedtuple("TaskInfo", ["name", "system", "task", "init_obs", 
    "dynamics", "perf_metric", "gen_sysid_trajs", "gen_surr_trajs"])

halfcheetah = ampc.System([f"x{i}" for i in range(18)], [f"u{i}" for i in range(6)])
env = gym.make("HalfCheetah-v2")
halfcheetah.dt = env.dt

def load_buffer(system, env_name="HalfCheetah-v2", buffer_dir="buffers", 
        prefix=None):
    if prefix is None:
        prefix = "Robust_" + env_name + "_0_"
    states = np.load(os.path.join(buffer_dir, prefix+"state.npy"))
    actions = np.load(os.path.join(buffer_dir, prefix+"action.npy"))
    next_states = np.load(os.path.join(buffer_dir, prefix+"next_state.npy"))

    def gym_to_obs(gym):
        #return np.concatenate([[0], gym])
        return gym

    episode_start = 0
    trajs = []
    for i in range(states.shape[0]):
        if i == states.shape[0]-1 or (next_states[i] != states[i+1]).any():
            traj = ampc.empty(system, i - episode_start + 1)
            traj.obs[:] = np.apply_along_axis(gym_to_obs, 1, 
                    states[episode_start:i+1])
            traj.ctrls[:] = actions[episode_start:i+1]
            trajs.append(traj)
            episode_start = i+1
    return trajs

def halfcheetah_dynamics(x, u, n_frames=5):
    old_state = env.sim.get_state()
    old_qpos = old_state[1]
    qpos = x[:len(old_qpos)]
    qvel = x[len(old_qpos):]
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
            old_state.act, old_state.udd_state)
    env.sim.set_state(new_state)
    #env.sim.forward()
    env.sim.data.ctrl[:] = u
    for _ in range(n_frames):
        env.sim.step()
    new_qpos = env.sim.data.qpos
    new_qvel = env.sim.data.qvel

    return np.concatenate([new_qpos, new_qvel])

def perf_metric(traj):
    cum_reward = 0.0
    for i in range(len(traj)-1):
        reward_ctrl = -0.1 * np.square(traj[i].ctrl).sum()
        reward_run = (traj[i+1, "x0"] - traj[i, "x0"]) / env.dt
        cum_reward += reward_ctrl + reward_run
    return 200 - cum_reward


def halfcheetah_task_buffer(buff=1):
    system = halfcheetah
    env = gym.make("HalfCheetah-v2")
    Q = np.eye(system.obs_dim)
    R = np.eye(system.ctrl_dim)
    F = np.eye(system.obs_dim)
    cost = QuadCost(system, Q, R, F)
    task = Task(system)
    task.set_cost(cost)
    task.set_ctrl_bounds(env.action_space.low, env.action_space.high)
    init_obs = np.concatenate([env.init_qpos, env.init_qvel])
    if buff == 1:
        buffer_dir = "buffers"
    elif buff == 2:
        buffer_dir = "buffers2"
    trajs = load_buffer(halfcheetah, "HalfCheetah-v2", buffer_dir)
    gen_sysid_trajs = lambda seed: trajs[:len(trajs)//2]
    gen_surr_trajs = lambda seed: trajs[len(trajs)//2:]
    return TaskInfo(name="HalfCheetah-Swingup",
            system=system, 
            task=task, 
            init_obs=init_obs, 
            dynamics=halfcheetah_dynamics,
            perf_metric=perf_metric,
            gen_sysid_trajs=gen_sysid_trajs,
            gen_surr_trajs=gen_surr_trajs)


def init_task(task):
    if task == "cartpole-swingup":
        return cartpole_swingup_task()
    else:
        raise
