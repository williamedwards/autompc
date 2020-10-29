# Created by William Edwards (wre@illinois.edu), 2020-10-21

# Standard library includes
import os, sys
from pdb import set_trace
import argparse

# External projects include
import numpy as np
import numpy.linalg as la
import gym
#import gym_cartpole_swingup
import custom_gym_cartpole_swingup
from custom_gym_cartpole_swingup.envs.cartpole_swingup import State as GymCartpoleState

# Internal project includes
import autompc as ampc
from autompc.tasks import Task, QuadCost

from collections import namedtuple

TaskInfo = namedtuple("TaskInfo", ["name", "system", "task", "init_obs", 
    "dynamics", "perf_metric", "gen_sysid_trajs", "gen_surr_trajs"])

cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
cartpole.dt = 0.05

def cartpole_simp_dynamics(y, u, g = 9.8, m = 1, L = 1, b = 0.1):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    theta, omega, x, dx = y
    return np.array([omega,
            g * np.sin(theta)/L - b * omega / (m*L**2) + u * np.cos(theta)/L,
            dx,
            u])

def dt_cartpole_dynamics(y,u,dt,g=9.8,m=1,L=1,b=1.0):
    y += dt * cartpole_simp_dynamics(y,u[0],g,m,L,b)
    return y

def gen_trajs(traj_len, num_trajs, dt, seed=42,
        init_min = [-1.0, 0.0, 0.0, 0.0], init_max=[1.0, 0.0, 0.0, 0.0],
        umin=-20.0, umax=20.0):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(num_trajs):
        state0 = [rng.uniform(minval, maxval, 1)[0] for minval, maxval 
                in zip(init_min, init_max)]
        y = state0[:]
        traj = ampc.zeros(cartpole, traj_len)
        traj.obs[:] = y
        for i in range(traj_len):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 1)
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

def cartpole_swingup_task():
    system = cartpole
    Q = np.eye(4)
    R = np.eye(1)
    F = np.eye(4)
    cost = QuadCost(system, Q, R, F)
    task = Task(system)
    task.set_cost(cost)
    task.set_ctrl_bound("u", -20.0, 20.0)
    init_obs = np.array([3.1, 0.0, 0.0, 0.0])
    def perf_metric(traj, threshold=0.2):
        cost = 0.0
        for i in range(len(traj)):
            if (np.abs(traj[i].obs[0]) > threshold 
                    or np.abs(traj[i].obs[1]) > threshold):
                cost += 1
        return cost
    def dynamics(y, u):
        return dt_cartpole_dynamics(y, u, system.dt)
    init_max = np.array([1.0, 10.0, 1.0, 10.0])
    gen_sysid_trajs = lambda seed: gen_trajs(200, 500, dt=system.dt,
            init_max=init_max, init_min=-init_max, seed=seed)
    gen_surr_trajs = lambda seed: gen_trajs(200, 500, dt=system.dt,
            init_max=init_max, init_min=-init_max, seed=seed)
    return TaskInfo(name="CartPole-Swingup",
            system=system, 
            task=task, 
            init_obs=init_obs, 
            dynamics=dynamics,
            perf_metric=perf_metric,
            gen_sysid_trajs=gen_sysid_trajs,
            gen_surr_trajs=gen_surr_trajs)


def init_task(task):
    if task == "cartpole-swingup":
        return cartpole_swingup_task()
    else:
        raise
