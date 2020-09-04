"""Just test if my MLP code is functioning as expected
"""
import argparse
import numpy as np
from joblib import Memory

import autompc as ampc
from test_shared import *


memory = Memory("cache")
linsys = ampc.System(['x', 'v'], ['a'])
linsys.dt = 0.1
pendulum = ampc.System(["ang", "angvel"], ["torque"])
pendulum.dt = 0.05
cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
cartpole.dt = 0.05
planar_drone = ampc.System(["x", "dx", "y", "dy", "theta", "omega"], ["u1", "u2"])
planar_drone = 0.05


def test_cartpole():
    """Test the MLP model and potentially cem on cartpole problem"""
    from autompc.sysid import MLP
    # collect trajectories
    dt = 0.05
    umin, umax = -2, 2
    trajs = collect_cartpole_trajs(dt, umin, umax, num_trajs=10)
    n_hidden, hidden_size, nonlintype, n_iter, n_batch, lr = 2, 32, 'relu', 10, 64, 1e-3
    model = MLP(cartpole, n_hidden, hidden_size, nonlintype, n_iter, n_batch, lr)
    model.train(trajs)
    # make predictions...
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([np.pi, 0, 0, 0])
    sim_traj[0].obs[:] = x
    us = []

    for step in range(200):
        u = np.random.random(1)
        x = dt_cartpole_dynamics(sim_traj[-1].obs, u, dt)
        newx = model.pred(sim_traj[-1].obs, u)
        newx, jx, ju = model.pred_diff(sim_traj[-1].obs, u)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        us.append(u)


if __name__ == '__main__':
    test_cartpole()
    return
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['pendulum', 'adaptive_pendulum',
        "cartpole", 'adaptive_cartpole'], default='pendulum', help='Specify which system id to test')
    args = parser.parse_args()
    if args.model == 'pendulum':
        test_pendulum()
    if args.model == 'adaptive_pendulum':
        test_adaptive_pendulum()
    if args.model == 'cartpole':
        test_cartpole()
    if args.model == 'adaptive_cartpole':
        test_adaptive_cartpole()