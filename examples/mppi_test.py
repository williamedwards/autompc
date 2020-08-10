"""
Test the nmpc.py file.
"""
import argparse
import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pdb import set_trace

from autompc.sysid.dummy_nonlinear import DummyNonlinear
from autompc.control import MPPI
from autompc import Task

from joblib import Memory
from scipy.integrate import solve_ivp

def animate_pendulum(fig, ax, dt, traj):
    ax.grid()

    line, = ax.plot([0.0, 0.0], [0.0, -1.0], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([0.0, 0.0], [0.0, -1.0])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        #i = min(i, ts.shape[0])
        line.set_data([0.0, np.sin(traj[i,"ang"]+np.pi)], 
                [0.0, -np.cos(traj[i,"ang"] + np.pi)])
        time_text.set_text('t={:2f}'.format(dt*i))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani

def animate_cartpole(fig, ax, dt, traj):
    ax.grid()

    line, = ax.plot([0.0, 0.0], [0.0, -1.0], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ctrl_text = ax.text(0.7, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([0.0, 0.0], [0.0, -1.0])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        #i = min(i, ts.shape[0])
        line.set_data([traj[i,"x"], traj[i,"x"]+np.sin(traj[i,"theta"]+np.pi)], 
                [0, -np.cos(traj[i,"theta"] + np.pi)])
        time_text.set_text('t={:.2f}'.format(dt*i))
        ctrl_text.set_text("u={:.3e}".format(traj[i,"u"]))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani


class Pendulum:
    def __init__(self):
        self.dt = 0.05
        self.g = 9.8
        self.m = 1
        self.L = 1
        self.b = 0.1

    def pred(self, state, ctrl):
        theta, omega = state
        g, m, L, b = self.g, self.m, self.L, self.b
        x0 = np.array([theta + omega * self.dt, omega + self.dt * ((ctrl[0] - b * omega) / (m * L ** 2) - g * np.sin(theta) / L)])
        # x1 = dt_pendulum_dynamics(state, ctrl, self.dt)
        return x0 + np.random.normal(scale=0.05, size=2)

    def pred_parallel(self, state, ctrl):
        theta, omega = state.T
        g, m, L, b = self.g, self.m, self.L, self.b
        return np.c_[theta + omega * self.dt, omega + self.dt * ((ctrl[:, 0] - b * omega) / (m * L ** 2) - g * np.sin(theta) / L)]

    def traj_to_state(self, traj):
        return traj.obs[-1]


def lqr_task_to_mppi_cost(task, dt):
    Q, R, F = task._quad_cost
    if F is None:
        terminal_cost = None
    else:
        terminal_cost = lambda x: np.einsum('ij,jk,ik->i', x, F, x)
    path_cost = lambda x, u: dt * (np.einsum('ij,jk,ik->i', x, Q, x) + np.einsum('ij,jk,ik->i', u, R, u))
    return path_cost, terminal_cost


class CartPole:
    def __init__(self):
        self.dt = 0.1

    def pred(self, state, ctrl, g = 9.8, m_c = 1, m_p = 1, L = 1, b = 1.0):
        # return dt_cartpole_dynamics(state, ctrl, self.dt)  # TODO: see why RK4 fails...
        theta, omega, x, dx = state
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

    def pred_parallel(self, state, ctrl, g = 9.8, m_c = 1, m_p = 1, L = 1, b = 1.0):
        theta, omega, x, dx = state.T  # TODO: implement parallel version of RK4...
        u = ctrl[:, 0]
        statedot = np.c_[omega,
                1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                    - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                    - (m_c+m_p+m_p)*g*np.sin(theta)
                    - b*omega),
                dx,
                1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                    (L*omega**2 + g*np.cos(theta)))]
        return state + self.dt * statedot

    def traj_to_state(self, traj):
        return traj.obs[-1]


# Here I test the discrete sindy example (maybe continuous is better to be honest...)
def test_pendulum():
    model = Pendulum()
    model.system = pendulum
    # Now it's time to apply the controller
    task1 = Task(pendulum)
    Q = np.diag([100.0, 1.0])
    R = np.diag([1.0]) 
    F = 100 * np.eye(2)
    task1.set_quad_cost(Q, R, F)
    path_cost, term_cost = lqr_task_to_mppi_cost(task1, model.dt)
    nmpc = MPPI(model.pred_parallel, path_cost, term_cost, model, H=15, sigma=10, num_path=100)
    # just give a random initial state
    sim_traj = ampc.zeros(pendulum, 1)
    x = np.array([np.pi, 0])
    sim_traj[0].obs[:] = x

    for _ in range(200):
        u, _ = nmpc.run(sim_traj)
        x = model.traj_to_state(sim_traj)
        x = model.pred(x, u)
        print('x = ', x, 'u = ', u)
        # x = dt_pendulum_dynamics(x, u, dt)
        sim_traj[-1, "torque"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
    print(sim_traj.obs)
    raise
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ani = animate_pendulum(fig, ax, model.dt, sim_traj)
    plt.show()

def test_cartpole():
    # Now it's time to apply the controller
    task1 = Task(cartpole)
    Q = np.diag([500.0, 15.0, 10.0, 1.0])
    R = np.diag([0.1]) 
    F = np.diag([10., 10., 2., 10.])
    task1.set_quad_cost(Q, R, F)
    model = CartPole()
    model.system = cartpole
    path_cost, term_cost = lqr_task_to_mppi_cost(task1, model.dt)
    nmpc = MPPI(model.pred_parallel, path_cost, term_cost, model, H=20, sigma=5, num_path=100)
    # just give a random initial state
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([0.1, 0., 0, 0])
    sim_traj[0].obs[:] = x
    us = []

    for step in range(100):
        u, _ = nmpc.run(sim_traj)
        #u = -np.zeros((1,))
        x = model.pred(x, u)
        print('state = ', x, ' u = ', u)
        sim_traj.ctrls[-1] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        us.append(u)
    print('states = ', sim_traj.obs)
    raise
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ani = animate_cartpole(fig, ax, dt, sim_traj)
    #plt.show()
    ani.save("out/nmpc_test/aug05_04.mp4")


pendulum = ampc.System(["ang", "angvel"], ["torque"])
cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])

def pendulum_dynamics(y,u,g=9.8,m=1,L=1,b=0.1):
    theta, omega = y
    return [omega,((u[0] - b*omega)/(m*L**2)
        - g*np.sin(theta)/L)]

def dt_pendulum_dynamics(y,u,dt,g=9.8,m=1,L=1,b=0.1):
    y[0] += np.pi
    sol = solve_ivp(lambda t, y: pendulum_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y[0] -= np.pi
    y = sol.y.reshape((2,))
    y[0] -= np.pi
    return y

def collect_pendulum_trajs(dt, umin, umax):
    # Generate trajectories for training
    num_trajs = 100
    trajs = gen_pendulum_trajs(dt, num_trajs, umin, umax)
    return trajs


def cartpole_dynamics(y, u, g = 9.8, m_c = 1, m_p = 1, L = 1, b = 1.0):
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
    #return [omega,
    #        g * np.sin(theta)/L - b * omega / (m*L**2) + u * np.sin(theta)/L,
    #        dx,
    #        u]
    return [omega,
            1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                - (m_c+m_p+m_p)*g*np.sin(theta)
                - b*omega),
            dx,
            1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                (L*omega**2 + g*np.cos(theta)))]

def dt_cartpole_dynamics(y,u,dt,g=9.8,m=1,L=1,b=1.0):
    y[0] += np.pi
    sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y[0] -= np.pi
    out = sol.y.reshape((4,))
    out[0] -= np.pi
    return out

def collect_cartpole_trajs(dt, umin, umax):
    # Generate trajectories for training
    num_trajs = 100
    trajs = gen_cartpole_trajs(dt, num_trajs, umin, umax)
    return trajs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['pendulum', 
        "cartpole"], default='pendulum', help='Specify which system id to test')
    args = parser.parse_args()
    if args.model == 'pendulum':
        test_pendulum()
    if args.model == 'cartpole':
        test_cartpole()
