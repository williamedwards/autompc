# Created by William Edwards (wre@illinois.edu), 2020-10-21

# Standard library includes
import os, sys
from pdb import set_trace
import argparse

# External projects include
import numpy as np
from scipy.integrate import solve_ivp

# Internal project includes
import autompc as ampc
from autompc.tasks import Task, QuadCost

from collections import namedtuple

TaskInfo = namedtuple("TaskInfo", ["name", "system", "task", "init_obs", 
    "dynamics", "perf_metric", "gen_sysid_trajs", "gen_surr_trajs"])

acrobot = ampc.System(["theta1", "theta2", "dtheta1", "dtheta2"], ["u"])
acrobot.dt = 0.05

def acrobot_dynamics(y,u,m1=1,m2=1,l1=1,lc1=0.5,lc2=0.5,I1=1,I2=1,g=9.8):
    cos = np.cos
    sin = np.sin
    pi = np.pi
    theta1 = y[0]
    theta2 = y[1]
    dtheta1 = y[2]
    dtheta2 = y[3]
    d1 = m1 * lc1 ** 2 + m2 * \
        (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
    phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.)
    phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2) \
           - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)  \
        + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) + phi2
    # the following line is consistent with the java implementation and the
    # book
    ddtheta2 = (u + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) \
        / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2])


def dt_acrobot_dynamics(y,u,dt):
    y = np.copy(y)
    y[0] += np.pi
    y += dt * acrobot_dynamics(y,u[0])
    y[0] -= np.pi
    return y

def gen_trajs(traj_len, num_trajs, dt, seed=42,
        init_min = [-1.0, 0.0, 0.0, 0.0], init_max=[1.0, 0.0, 0.0, 0.0],
        umin=-2.0, umax=2.0):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(num_trajs):
        state0 = [rng.uniform(minval, maxval, 1)[0] for minval, maxval 
                in zip(init_min, init_max)]
        y = state0[:]
        traj = ampc.zeros(acrobot, traj_len)
        traj.obs[:] = y
        for i in range(traj_len):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 1)
            y = dt_acrobot_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

def acrobot_swingup_task():
    system = acrobot
    Q = np.eye(4)
    R = np.eye(1)
    F = np.eye(4)
    cost = QuadCost(system, Q, R, F)
    task = Task(system)
    task.set_cost(cost)
    task.set_ctrl_bound("u", -1.0, 1.0)
    init_obs = np.array([3.1, 0.0, 0.0, 0.0])
    def perf_metric(traj):
        return cost(traj)
    def dynamics(y, u):
        return dt_acrobot_dynamics(y, u, acrobot.dt)
    init_max = np.array([1.0, 1.0, 3.0, 3.0])
    gen_sysid_trajs = lambda seed: gen_trajs(200, 500, dt=system.dt,
            init_max=init_max, init_min=-init_max, seed=seed)
    gen_surr_trajs = lambda seed: gen_trajs(200, 500, dt=system.dt,
            init_max=init_max, init_min=-init_max, seed=seed)
    return TaskInfo(name="Acrobot-Swingup",
            system=system, 
            task=task, 
            init_obs=init_obs, 
            dynamics=dynamics,
            perf_metric=perf_metric,
            gen_sysid_trajs=gen_sysid_trajs,
            gen_surr_trajs=gen_surr_trajs)

