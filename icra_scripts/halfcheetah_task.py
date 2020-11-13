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


@memory.cache
def gen_trajs(num_trajs=1000, traj_len=1000, seed=42):
    rng = np.random.default_rng(seed)
    trajs = []
    env.seed(int(rng.integers(1 << 30)))
    env.action_space.seed(int(rng.integers(1 << 30)))
    for i in range(num_trajs):
        init_obs = env.reset()
        traj = ampc.zeros(halfcheetah, traj_len)
        traj[0].obs[:] = np.concatenate([[0], init_obs])
        for j in range(1, traj_len):
            action = env.action_space.sample()
            traj[j-1].ctrl[:] = action
            #obs, reward, done, info = env.step(action)
            obs = halfcheetah_dynamics(traj[j-1].obs[:], action)
            traj[j].obs[:] = obs
        trajs.append(traj)
    return trajs

def halfcheetah_task():
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
    gen_sysid_trajs = lambda seed: gen_trajs(500, 200, seed=seed)
    gen_surr_trajs = lambda seed: gen_trajs(500, 200,  seed=seed)
    return TaskInfo(name="CartPole-Swingup",
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
