"""
Test the nmpc.py file.
"""
import argparse
import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt

from autompc.sysid.dummy_nonlinear import DummyNonlinear
from autompc.control import NonLinearMPC
from autompc import Task

from joblib import Memory
from scipy.integrate import solve_ivp


def test_dummy():
    dummy_sys = ampc.System(["x1", "x2"], ["u"])
    dummy_model = DummyNonlinear(dummy_sys)

    task1 = Task(dummy_sys)
    Q = np.eye(2)
    R = np.eye(1)
    F = 10 * np.eye(2)
    task1.set_quad_cost(Q, R, F)

    horizon = 8
    nmpc = NonLinearMPC(dummy_sys, dummy_model, task1, horizon)
    # just give a random initial state
    sim_traj = ampc.zeros(dummy_sys, 1)
    x = np.array([2, 1.0])
    sim_traj[0].obs[:] = x

    for _ in range(10):
        u, _ = nmpc.run(sim_traj)
        print('u = ', u)
        x = dummy_model.traj_to_state(sim_traj)
        x = dummy_model.pred(x, u)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])

    print(sim_traj[:, "x1"], sim_traj[:, 'x2'])
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.plot(sim_traj[:, 'x1'], sim_traj[:, 'x2'])
    plt.show()


# Here I test the discrete sindy example (maybe continuous is better to be honest...)
def test_sindy():
    from autompc.sysid import SINDy
    cs = SINDy.get_configuration_space(pendulum)
    s = cs.get_default_configuration()
    s["trig_basis"] = "true"
    s["trig_freq"] = 1
    s["poly_basis"] = "false"
    trajs = collect_pendulum_trajs()
    model = ampc.make_model(pendulum, SINDy, s)
    model.train(trajs)
    # Now it's time to apply the controller
    task1 = Task(pendulum)
    Q = np.eye(2)
    R = np.eye(1)
    F = 10 * np.eye(2)
    task1.set_quad_cost(Q, R, F)

    horizon = 8  # this is indeed too short for a frequency of 100 Hz model
    nmpc = NonLinearMPC(pendulum, model, task1, horizon)
    # just give a random initial state
    sim_traj = ampc.zeros(pendulum, 1)
    x = np.array([-np.pi, 0])
    sim_traj[0].obs[:] = x

    for _ in range(10):
        u, _ = nmpc.run(sim_traj)
        print('u = ', u)
        x = model.traj_to_state(sim_traj)
        x = model.pred(x, u)
        sim_traj[-1, "torque"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])

    print(sim_traj[:, "ang"], sim_traj[:, 'angvel'])
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.plot(sim_traj[:, 'ang'], sim_traj[:, 'angvel'])
    plt.show()


memory = Memory("cache")

pendulum = ampc.System(["ang", "angvel"], ["torque"])

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


def collect_pendulum_trajs():
    dt = 0.01

    umin = -2.0
    umax = 2.0

    # Generate trajectories for training
    num_trajs = 100
    trajs = gen_trajs(dt, num_trajs, umin, umax)
    return trajs


@memory.cache
def gen_trajs(dt, num_trajs, umin, umax):
    rng = np.random.default_rng(42)
    trajs = []
    for _ in range(num_trajs):
        y = [-np.pi, 0.0]
        traj = ampc.zeros(pendulum, 400)
        for i in range(400):
            traj[i].obs[:] = y
            u = rng.uniform(umin, umax, 1)
            y = dt_pendulum_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['koopman', 'sindy'], help='Specify which system id to test')
    args = parser.parse_args()
    if args.model == 'koopman':
        test_dummy()
    if args.model == 'sindy':
        test_sindy()