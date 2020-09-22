"""
Test how mlp controller works
"""
import argparse
import numpy as np
from joblib import Memory

import autompc as ampc
from autompc.sysid import MLP
from autompc.control import CEM, MPPI
from test_shared import *
from mppi_test import lqr_task_to_mppi_cost


pendulum.dt = 0.05
cartpole.dt = 0.05
planar_drone.dt = 0.05


def test_cartpole(control='mppi'):
    """Test the MLP model and potentially cem on cartpole problem"""
    from autompc.sysid import MLP
    # collect trajectories
    dt = 0.05
    umin, umax = -2, 2
    trajs = collect_cartpole_trajs(dt, umin, umax, num_trajs=200)
    n_hidden, hidden_size, nonlintype, n_iter, n_batch, lr = 2, 32, 'relu', 10, 64, 1e-3
    model = MLP(cartpole, n_hidden, hidden_size, nonlintype, n_iter, n_batch, lr)
    model.train(trajs)
    # make predictions...
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([np.pi, 0, 0, 0])
    sim_traj[0].obs[:] = x
    us = []
    # define task
    task1 = ampc.Task(cartpole)
    Q = np.diag([10.0, 5.0, 50.0, 50.0])
    R = np.diag([0.3]) 
    F = np.diag([10., 10., 10., 10.]) * 10
    task1.set_quad_cost(Q, R, F)
    path_cost, term_cost = lqr_task_to_mppi_cost(task1, cartpole.dt)
    # construct controller
    if control == 'mppi':
        nmpc = MPPI(model.pred_parallel, path_cost, term_cost, model, H=15, sigma=10, num_path=200)
    elif control == 'cem':
        nmpc = CEM(model.pred_parallel, path_cost, term_cost, model, H=15, sigma=1, num_path=1500)

    for step in range(200):
        u = np.random.random(1)
        newx = model.pred(sim_traj[-1].obs, u)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        us.append(u)
        if np.linalg.norm(newx) < 5e-2:
            break
    print('final state is ', newx)


if __name__ == '__main__':
    test_cartpole()
    raise
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