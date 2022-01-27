# Created by William Edwards (wre2@illinois.edu), 2021-01-09

# Standard library includes
import sys
from pdb import set_trace

# External library includes
import numpy as np
import autompc as ampc

# Project includes

def uniform_random_generate(system, ocp, dynamics, rng, init_min, init_max, 
        traj_len, n_trajs):
    trajs = []
    for _ in range(n_trajs):
        state0 = [rng.uniform(minval, maxval, 1)[0] for minval, maxval 
                in zip(init_min, init_max)]
        y = state0[:]
        traj = ampc.zeros(system, traj_len)
        traj.obs[:] = y
        umin, umax = ocp.get_ctrl_bounds().T
        for i in range(traj_len):
            traj[i].obs[:] = y
            u = rng.uniform(umin, umax, system.ctrl_dim)
            y = dynamics(y, u)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

def prbs_generate(system, ocp, dynamics, rng, init_min, init_max,
        traj_len, n_trajs, states, Nswitch):
    trajs = []
    for _ in range(n_trajs):
        # Compute control sequence
        switches = rng.choice(traj_len, Nswitch)  
        switches = np.concatenate([[0], switches, [traj_len]])
        u = np.zeros(traj_len) 
        for ps, ns in zip(switches[:-1], switches[1:]):
            u[ps:ns] = rng.choice(states)

        state0 = [rng.uniform(minval, maxval, 1)[0] for minval, maxval 
                in zip(init_min, init_max)]
        y = state0[:]
        traj = ampc.zeros(system, traj_len)
        traj.obs[:] = y
        for i in range(traj_len):
            traj[i].obs[:] = y
            y = dynamics(y, u[i])
            traj[i].ctrl[:] = u[i]
        trajs.append(traj)
    return trajs

def random_walk_generate(system, ocp, dynamics, rng, init_min, init_max, walk_rate,
        traj_len, n_trajs):
    trajs = []
    for _ in range(n_trajs):
        state0 = [rng.uniform(minval, maxval, 1)[0] for minval, maxval 
                in zip(init_min, init_max)]
        y = state0[:]
        traj = ampc.zeros(system, traj_len)
        traj.obs[:] = y
        umin, umax = ocp.get_ctrl_bounds().T
        uamp = np.min([umin, umax])
        u = rng.uniform(umin, umax, system.ctrl_dim)
        step_size = walk_rate * system.dt
        for i in range(traj_len):
            traj[i].obs[:] = y
            u += uamp * step_size * rng.uniform(-1, 1, system.ctrl_dim)
            u = np.clip(u, umin, umax)
            y = dynamics(y, u)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs


def periodic_control_generate(system, ocp, dynamics, rng, init_min, init_max, U_1, 
        traj_len, n_trajs):
    trajs = []
    periods = list(range(1, traj_len, max([1, traj_len // n_trajs])))
    print("periods=", periods)
    for period in periods:
        state0 = [rng.uniform(minval, maxval, 1)[0] for minval, maxval 
                in zip(init_min, init_max)]
        y = state0[:]
        traj = ampc.zeros(system, traj_len)
        traj.obs[:] = y
        umin, umax = ocp.get_ctrl_bounds().T
        uamp = np.min([umin, umax])
        for i in range(traj_len):
            traj[i].obs[:] = y
            u  = uamp * U_1 * np.cos(2 * np.pi * i / period)
            y = dynamics(y, u)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

def multisine_generate(system, task, dynamics, rng, init_min, init_max, n_freqs,
        traj_len, n_trajs, abort_if=None):
    trajs = []
    periods  = list(range(1, traj_len, n_freqs))
    umin, umax = task.get_ctrl_bounds().T
    uamp = (umax - umin) / 2
    umed = (umax + umin) / 2
        
    for _ in range(n_trajs):
        weights = []
        for i in range(system.ctrl_dim):
            vals = rng.uniform(size=len(periods)-1)
            vals = np.concatenate([[0.0], np.sort(vals), [1.0]])
            weight = vals[1:] - vals[:-1]
            weights.append(weight)
        weights = np.array(weights)
        phases = rng.uniform(0, 2*np.pi, len(periods))

        state0 = [rng.uniform(minval, maxval, 1)[0] for minval, maxval 
                in zip(init_min, init_max)]
        y = state0[:]
        traj = ampc.zeros(system, traj_len)
        traj.obs[:] = y
        umin, umax = task.get_ctrl_bounds().T
        for i in range(traj_len):
            traj[i].obs[:] = y
            u = np.zeros(system.ctrl_dim)
            for j, period in enumerate(periods):
                u += weights[:,j] * np.cos(2 * np.pi * i / period + phases[j])
            u = uamp * u + umed
            y = dynamics(y, u)
            traj[i].ctrl[:] = u
            if not abort_if is None and abort_if(y):
                traj = traj[:i]
                break
        trajs.append(traj)
    return trajs
