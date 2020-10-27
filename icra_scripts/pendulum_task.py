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

pendulum = ampc.System(["ang", "angvel"], ["torque"])
pendulum.dt = 0.05

def pendulum_dynamics(y,u,g=9.8,m=1,L=1,b=0.1):
    theta, omega = y
    return [omega,((u[0] - b*omega)/(m*L**2)
        - g*np.sin(theta)/L)]

def dt_pendulum_dynamics(y,u,dt,g=9.8,m=1,L=1,b=0.1):
    y[0] += np.pi
    sol = solve_ivp(lambda t, y: pendulum_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y = sol.y.reshape((2,))
    y[0] -= np.pi
    return sol.y.reshape((2,))

def gen_trajs(traj_len, num_trajs, dt, seed=42,
        init_min = [-1.0, 0.0], init_max=[1.0, 0.0],
        umin=-2.0, umax=2.0):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(num_trajs):
        state0 = [rng.uniform(minval, maxval, 1)[0] for minval, maxval 
                in zip(init_min, init_max)]
        y = state0[:]
        traj = ampc.zeros(pendulum, traj_len)
        traj.obs[:] = y
        for i in range(traj_len):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 1)
            y = dt_pendulum_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

def pendulum_swingup_task():
    system = pendulum
    Q = np.eye(2)
    R = np.eye(1)
    F = np.eye(2)
    cost = QuadCost(system, Q, R, F)
    task = Task(system)
    task.set_cost(cost)
    task.set_ctrl_bound("torque", -1.0, 1.0)
    init_obs = np.array([3.1, 0.0])
    def perf_metric(traj):
        return cost(traj)
    def dynamics(y, u):
        return dt_pendulum_dynamics(y, u, pendulum.dt)
    init_max = np.array([1.0, 3.0])
    gen_sysid_trajs = lambda seed: gen_trajs(200, 500, dt=system.dt,
            init_max=init_max, init_min=-init_max, seed=seed)
    gen_surr_trajs = lambda seed: gen_trajs(200, 500, dt=system.dt,
            init_max=init_max, init_min=-init_max, seed=seed)
    return TaskInfo(name="Pendulum-Swingup",
            system=system, 
            task=task, 
            init_obs=init_obs, 
            dynamics=dynamics,
            perf_metric=perf_metric,
            gen_sysid_trajs=gen_sysid_trajs,
            gen_surr_trajs=gen_surr_trajs)

