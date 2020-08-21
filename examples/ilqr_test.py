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
from autompc.control import IterativeLQR, NonLinearMPC
from autompc import Task

from joblib import Memory
from scipy.integrate import solve_ivp

from test_shared import *


linsys = ampc.System(['x', 'v'], ['a'])
linsys.dt = 0.1
pendulum = ampc.System(["ang", "angvel"], ["torque"])
pendulum.dt = 0.05
cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
cartpole.dt = 0.05
planar_drone = ampc.System(["x", "dx", "y", "dy", "theta", "omega"], ["u1", "u2"])
planar_drone = 0.05


class CartPole(ampc.Model):
    def __init__(self):
        super().__init__(cartpole)
        g = 9.8; m_c = 1; m_p = 0.1; L = 1; b = 1.00; dt = cartpole.dt

        import autograd.numpy as np
        from autograd import jacobian

        def dynamics(xin):
            theta0, omega, x, dx, u = xin
            theta = theta0 + np.pi
            xdot = np.array([omega,
                1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                    - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                    - (m_c+m_p+m_p)*g*np.sin(theta)
                    - b*omega),
                dx,
                1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                    (L*omega**2 + g*np.cos(theta)))])
            return xin[:4] + xdot * dt

        self._dyn_fun = dynamics
        self._dyn_jac = jacobian(dynamics)

    def traj_to_state(self, traj):
        return traj.obs[-1]

    def pred(self, state, ctrl):
        xin = np.concatenate((state, ctrl))
        return self._dyn_fun(xin)
    
    def pred_diff(self, state, ctrl):
        xin = np.concatenate((state, ctrl))
        x = self._dyn_fun(xin)
        jac = self._dyn_jac(xin)
        return x, jac[:, :4], jac[:, 4:]

    def get_configuration_space(self):
        pass
    
    def state_dim(self):
        pass
    
    def update_state(self, x, u, obs):
        return obs


def test_dummy_linear():
    model = DummyLinear(linsys, np.array([[1., linsys.dt], [0, 1.]]), np.array([[0.], [linsys.dt]]))
    task1 = Task(linsys)
    Q = np.diag([1., 1.])
    R = np.diag([0.01]) 
    F = np.diag([10.0, 10.0])
    task1.set_quad_cost(Q, R, F)
    # construct ilqr instance
    init_state = np.array([0.5, 0.5])
    ilqr = IterativeLQR(linsys, task1, model, 10)
    # start simulation
    sim_traj = ampc.zeros(linsys, 1)
    sim_traj[0].obs[:] = init_state
    us = []
    for step in range(400):
        state = ilqr.traj_to_state(sim_traj)
        u, _ = ilqr.run(state, None)
        print('u = ', u)
        # x = model.pred(x, u)
        newx = model.pred(state, u)
        sim_traj[-1].ctrl[:] = u
        sim_traj = ampc.extend(sim_traj, [newx], [[0.0]])

        us.append(u)
    print('states are ', sim_traj.obs)


def test_cartpole():
    """Do cartpole with true dynamics"""
    model = CartPole()
    task1 = Task(cartpole)
    Q = np.diag([1., 1., 1., 1.])
    R = np.diag([0.1]) 
    F = np.diag([10.0, 10.0, 10., 10.]) * 10
    task1.set_quad_cost(Q, R, F)
    # construct ilqr instance
    init_state = np.array([np.pi, 0.0, 0., 0.])
    # horizon_int = 40  # this works well with bound = 15
    horizon_int = 60
    ubound = np.array([[-5], [5]])
    ilqr = IterativeLQR(cartpole, task1, model, horizon_int, reuse_feedback=0, ubounds=ubound, mode='barrier')  # change to 0/None if want to reoptimize at every step
    # np.random.seed(42)
    ilqr._guess = np.random.uniform(-5, 5, horizon_int)[:, None]
    # nmpc = NonLinearMPC(cartpole, model, task1, horizon_int * cartpole.dt)
    # start simulation
    sim_traj = ampc.zeros(cartpole, 1)
    sim_traj[0].obs[:] = init_state
    us = []
    for step in range(200):
        state = model.traj_to_state(sim_traj)
        if np.linalg.norm(state) < 1e-3:
            break
        u, _ = ilqr.run(state, state)
        # u, _ = nmpc.run(state, state)
        # print(np.linalg.norm(ilqr._states[:horizon_int] - nmpc._guess[:horizon_int * 4].reshape((horizon_int, 4))))
        print('u = ', u)
        newx = model.pred(state, u)
        # newx = dt_cartpole_dynamics(state, dt)
        # newx += 0.01 * np.random.uniform(-np.ones(4), np.ones(4))
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
        fig.savefig('cartpole_ilqr_state.png')
    else:
        fig.savefig('cartpole_ilqr_state_bound_control_%f.png' % ubound[1, 0])


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
