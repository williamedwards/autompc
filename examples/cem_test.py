"""
Test the nmpc.py file.
"""
import argparse
import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from autompc.sysid.dummy_nonlinear import DummyNonlinear
from autompc.sysid.dummy_linear import DummyLinear
from autompc.control import IterativeLQR, NonLinearMPC, CEM
from autompc import Task

from joblib import Memory
from scipy.integrate import solve_ivp

from test_shared import *
from mppi_test import CartPole, lqr_task_to_mppi_cost


pendulum = ampc.System(["ang", "angvel"], ["torque"])
pendulum.dt = 0.05
cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
cartpole.dt = 0.05


class CartPole:
    def __init__(self, choice='Euler'):
        self.dt = cartpole.dt
        if choice == 'Euler':
            self.pred = self.pred_euler
        else:
            self.pred = self.pred_rk4

    def pred_euler(self, state, ctrl, g = 9.8, m_c = 1., m_p = 0.1, L = 1., b = 0.2):
        theta, omega, x, dx = state
        theta = theta + np.pi
        u = ctrl[0]
        statedot = np.array([omega,
                1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                    - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                    - (m_c+m_p+m_p)*g*np.sin(theta)
                    - b*omega),
                dx,
                1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                    (L*omega**2 + g*np.cos(theta)))])
        return state + self.dt * statedot

    def pred_rk4(self, state, ctrl):
        return dt_cartpole_dynamics(state, ctrl, self.dt)

    def pred_parallel(self, state, ctrl, g = 9.8, m_c = 1., m_p = 0.1, L = 1., b = 0.2):
        theta, omega, x, dx = state.T 
        theta = theta + np.pi
        u = ctrl[:, 0]
        sth = np.sin(theta)
        cth = np.cos(theta)
        statedot = np.c_[omega,
                1.0/(L*(m_c+m_p+m_p*sth**2))*(-u*cth
                    - m_p*L*omega**2*cth*sth
                    - (m_c+m_p+m_p)*g*sth
                    - b*omega),
                dx,
                1.0/(m_c + m_p*sth**2)*(u + m_p*sth*
                    (L*omega**2 + g*cth))]
        return state + self.dt * statedot

    def traj_to_state(self, traj):
        return traj.obs[-1]


def test_cartpole():
    """Do cartpole with true dynamics"""
    model = CartPole()
    model.system = cartpole
    task1 = Task(cartpole)
    Q = np.diag([1., 1., 1., 1.])
    R = np.diag([0.1]) 
    F = np.diag([10.0, 10.0, 10., 10.]) * 10
    task1.set_quad_cost(Q, R, F)
    # construct ilqr instance
    init_state = np.array([np.pi, 0.0, 0., 0.])
    horizon_int = 40  # this works well with bound = 15
    # horizon_int = 60
    ubound = np.array([[-15], [15]])
    path_cost, term_cost = lqr_task_to_mppi_cost(task1, model.dt)
    cem = CEM(model.pred_parallel, path_cost, term_cost, model, H=horizon_int, sigma=1., num_path=1500, ubounds=ubound)
    # start simulation
    sim_traj = ampc.zeros(cartpole, 1)
    sim_traj[0].obs[:] = init_state
    us = []
    for step in range(200):
        state = model.traj_to_state(sim_traj)
        if np.linalg.norm(state) < 1e-3:
            break
        u, _ = cem.run(sim_traj)
        print('u = ', u)
        newx = model.pred(state, u)
        sim_traj[-1].ctrl[:] = u
        sim_traj = ampc.extend(sim_traj, [newx], [[0.0]])
        us.append(u)

    print('states are ', sim_traj.obs)
    print('control is ', sim_traj.ctrls)
    fig, ax = plt.subplots(3, 2)
    ax = ax.reshape(-1)
    state_names = ['theta', 'omega', 'x', 'dx']
    times = np.arange(sim_traj.obs.shape[0]) * cartpole.dt
    for i in range(4):
        ax[i].plot(times, sim_traj.obs[:, i])
        ax[i].set_ylabel(state_names[i])
        ax[i].set_xlabel('Time [s]')
    ax[4].plot(times[:-1], sim_traj.ctrls[:-1])
    ax[4].set(xlabel='Time [s]', ylabel='u')
    fig.tight_layout()
    if ubound is None:
        fig.savefig('cartpole_cem_state.png')
    else:
        fig.savefig('cartpole_cem_state_bound_control_%.2f.png' % (ubound[1, 0]))


def test_sindy_cartpole():
    from autompc.sysid import SINDy
    cs = SINDy.get_configuration_space(cartpole)
    s = cs.get_default_configuration()
    s["trig_basis"] = "true"
    s["poly_basis"] = "false"
    dt = 0.05
    umin, umax = -2, 2
    trajs = collect_cartpole_trajs(dt, umin, umax)
    cartpole.dt = dt
    model = ampc.make_model(cartpole, SINDy, s)
    model.trig_interaction = True
    model.train(trajs)

    # Now it's time to apply the controller
    task1 = Task(cartpole)
    Q = np.diag([10.0, 1.0, 10.0, 1.0])
    R = np.diag([1.0]) 
    F = np.diag([10., 10., 10., 10.])
    task1.set_quad_cost(Q, R, F)

    hori = 40  # hori means integer horizon... how many steps...
    ilqr = IterativeLQR(cartpole, task1, model, hori, reuse_feedback=None)
    # just give a random initial state
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([1.0, 0, 0, 0])
    sim_traj[0].obs[:] = x
    us = []

    constate = ilqr.traj_to_state(sim_traj[:1])
    for step in range(200):
        u, constate = ilqr.run(constate, sim_traj[-1].obs)
        print('u = ', u, 'state = ', sim_traj[-1].obs)
        x = dt_cartpole_dynamics(sim_traj[-1].obs, u, dt)
        # x = model.pred(sim_traj[-1].obs, u)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        us.append(u)
    print(sim_traj.obs)


def check_ilqr_different_feedback():
    """Check to make sure ilqr gets different feedback matrix compared with solving discrete Riccati equations"""
    model = CartPole()
    task1 = Task(cartpole)
    Q = np.diag([1., 1., 1., 1.])
    R = np.diag([0.1]) 
    F = np.diag([10.0, 10.0, 10., 10.]) * 10
    task1.set_quad_cost(Q, R, F)
    # construct ilqr instance
    init_state = np.array([1, 0.0, 0., 0.])
    horizon_int = 20
    ilqr = IterativeLQR(cartpole, task1, model, horizon_int, reuse_feedback=1)
    converged, states, ctrls, Ks, ks = ilqr.compute_ilqr(init_state, np.zeros((horizon_int, cartpole.ctrl_dim)))
    # so I can linearize around those stuff and get As and Bs
    As, Bs = [], []
    for i in range(horizon_int):
        _, Jx, Ju = model.pred_diff(states[i], ctrls[i])
        As.append(Jx)
        Bs.append(Ju)
    # use ricatti iteration to compute feedbacks
    Q = Q * cartpole.dt
    R = R * cartpole.dt
    Pfinal = F
    Ps = [Pfinal]
    iKs = []
    for i in range(horizon_int):
        A, B = As[-1 - i], Bs[-1 - i]
        Pnew = Q + A.T @ Ps[-1] @ A - A.T @ Ps[-1] @ B @ np.linalg.solve(B.T @ Ps[-1] @ B + R, B.T) @ Ps[-1] @ A
        Knew = -np.linalg.solve(B.T @ Ps[-1] @ B + R, B.T @ Ps[-1] @ A)
        Ps.append(Pnew)
        iKs.append(Knew)
    # now compare K
    for K, iK in zip(Ks, iKs[::-1]):
        print(K - iK)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['koopman', 'sindy-pendulum', 
        "sindy-cartpole", "linear-cartpole", "sindy-planar-drone", 
        "true-dyn-cartpole", "true-dyn-planar-drone"], 
        default='sindy-cartpole', help='Specify which system id to test')
    args = parser.parse_args()
    # test_dummy_linear()
    test_cartpole()
    # test_sindy_cartpole()
