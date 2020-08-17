"""
Test the nmpc.py file.
"""
import argparse
import numpy as np
import torch
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pdb import set_trace

from autompc.sysid.dummy_nonlinear import DummyNonlinear
from autompc.control import MPPI, MPPIAdaptive
from autompc import Task

from joblib import Memory
from scipy.integrate import solve_ivp


memory = Memory("cache")
pendulum = ampc.System(["ang", "angvel"], ["torque"])
cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])

GRAVITY = 9.8
M_C = 1
M_P = 0.2
LEN = 1
BB = 1

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
    def __init__(self, choice='Euler'):
        self.dt = 0.05
        self.g = 9.8
        self.m = 1
        self.L = 1
        self.b = 0.1
        if choice == 'Euler':
            self.pred = self.pred_euler
        else:
            self.pred = self.pred_rk4

    def pred_euler(self, state, ctrl):
        theta, omega = state
        theta += np.pi
        g, m, L, b = self.g, self.m, self.L, self.b
        x0 = np.array([theta + omega * self.dt, omega + self.dt * ((ctrl[0] - b * omega) / (m * L ** 2) - g * np.sin(theta) / L)])
        x0[0] -= np.pi
        return x0

    def pred_rk4(self, state, ctrl):
        return dt_pendulum_dynamics(state, ctrl, self.dt)

    def pred_parallel(self, state, ctrl):
        theta, omega = state.T
        theta += np.pi
        g, m, L, b = self.g, self.m, self.L, self.b
        return np.c_[state[:, 0] + omega * self.dt, omega + self.dt * ((ctrl[:, 0] - b * omega) / (m * L ** 2) - g * np.sin(theta) / L)]

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
    def __init__(self, choice='Euler'):
        self.dt = 0.1
        if choice == 'Euler':
            self.pred = self.pred_euler
        else:
            self.pred = self.pred_rk4

    def pred_euler(self, state, ctrl, g = GRAVITY, m_c = M_C, m_p = M_P, L = LEN, b = BB):
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

    def pred_parallel(self, state, ctrl, g = GRAVITY, m_c = M_C, m_p = M_P, L = LEN, b = BB):
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


# Here I test the discrete sindy example (maybe continuous is better to be honest...)
def test_pendulum():
    model = Pendulum()
    model.system = pendulum
    # Now it's time to apply the controller
    task1 = Task(pendulum)
    Q = np.diag([10.0, 1.0])
    R = np.diag([1.0]) 
    F = 100 * np.eye(2)
    task1.set_quad_cost(Q, R, F)
    path_cost, term_cost = lqr_task_to_mppi_cost(task1, model.dt)
    nmpc = MPPI(model.pred_parallel, path_cost, term_cost, model, H=15, sigma=10, num_path=200)
    # the first step is to use existing code to collect enough state-transitions and learn the model

    # just give a random initial state
    sim_traj = ampc.zeros(pendulum, 1)
    x = np.array([-np.pi, 0])
    sim_traj[0].obs[:] = x

    for _ in range(200):
        u, _ = nmpc.run(sim_traj)
        x = model.traj_to_state(sim_traj)
        x = model.pred(x, u)
        print('x = ', x, 'u = ', u)
        sim_traj[-1, "torque"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
    print(sim_traj.obs)
    fig, ax = plt.subplots(1, 2)
    ynames = ['theta', 'omega']
    for i in range(2):
        ax[i].plot(sim_traj.obs[:, i])
        ax[i].set(xlabel='Step', ylabel=ynames[i])
    plt.show()
    fig = plt.figure()
    raise
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ani = animate_pendulum(fig, ax, model.dt, sim_traj)
    plt.show()


def test_adaptive_pendulum():
    """Just test the adaptive mppi on the pendulum problem.
    Now it supports model re-training so hopefully a few iterations sovle the problem.
    """
    model = Pendulum(choice='RK4')
    model.system = pendulum

    import torch
    network = torch.nn.Sequential(
        torch.nn.Linear(3, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2)
    ).double()

    # Now it's time to apply the controller
    task1 = Task(pendulum)
    Q = np.diag([100.0, 1.0])
    R = np.diag([1.0]) 
    F = 100 * np.eye(2)
    task1.set_quad_cost(Q, R, F)
    path_cost, term_cost = lqr_task_to_mppi_cost(task1, model.dt)

    mppi = MPPIAdaptive(network, path_cost, term_cost, model, H=15, sigma=10, num_path=100)
    TRAIN_EPOCH = 50
    # the first step is to collect some initial trajectories and train the network...
    trajs = collect_pendulum_trajs(model.dt, -2, 2)
    mppi.init_network(trajs, niter=TRAIN_EPOCH)

    init_state = np.array([-np.pi, 0])
    # now we can run a few iterations, each iteration is one episode
    num_iter = 50
    num_step = 200
    for itr in range(num_iter):
        traj = mppi.run_episode(init_state, num_step, update_config={'niter': TRAIN_EPOCH})
        print('itr = ', itr, 'final states = ', traj.obs[-10:])


def test_cartpole():
    # Now it's time to apply the controller
    task1 = Task(cartpole)
    Q = np.diag([10.0, 5.0, 50.0, 50.0])
    R = np.diag([0.3]) 
    F = np.diag([10., 10., 10., 10.]) * 10
    task1.set_quad_cost(Q, R, F)
    model = CartPole('Euler')
    model.system = cartpole
    path_cost, term_cost = lqr_task_to_mppi_cost(task1, model.dt)
    nmpc = MPPI(model.pred_parallel, path_cost, term_cost, model, H=30, sigma=5, num_path=1500)
    # just give a random initial state
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([np.pi, 0., 0, 0])
    sim_traj[0].obs[:] = x
    us = []

    for step in range(300):
        u, _ = nmpc.run(sim_traj)
        #u = -np.zeros((1,))
        x = model.pred(x, u)
        print('state = ', x, ' u = ', u)
        sim_traj.ctrls[-1] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        us.append(u)
    fig, ax = plt.subplots(2, 2)
    ax = ax.reshape(-1)
    ynames = ['theta', 'omega', 'x', 'vx']
    for i in range(4):
        ax[i].plot(sim_traj.obs[:, i])
        ax[i].set(xlabel='Step', ylabel=ynames[i])
    plt.show()
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    #plt.show()
    ani = animate_cartpole(fig, ax, model.dt, sim_traj)
    ani.save("out/nmpc_test/aug05_04.mp4")


def test_adaptive_cartpole():
    """Just test the adaptive mppi on the pendulum problem.
    Now it supports model re-training so hopefully a few iterations sovle the problem.
    """
    model = CartPole(choice='Euler')
    model.system = cartpole

    import torch
    network = torch.nn.Sequential(
        torch.nn.Linear(5, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 4)
    ).double().cuda()

    # Now it's time to apply the controller
    task1 = Task(cartpole)
    Q = np.diag([5, 5.0, 5.0, 5.0])
    R = np.diag([0.1]) 
    F = np.diag([10., 10., 10., 10.]) * 10
    task1.set_quad_cost(Q, R, F=None)
    path_cost, term_cost = lqr_task_to_mppi_cost(task1, model.dt)

    mppi = MPPIAdaptive(network, path_cost, term_cost, model, H=30, sigma=10, num_path=1500, umin=-10, umax=10)
    TRAIN_EPOCH = 50
    # the first step is to collect some initial trajectories and train the network...
    trajs = collect_cartpole_trajs(model.dt, -2, 2)
    mppi.init_network(trajs, niter=TRAIN_EPOCH, lr=5e-4)

    init_state = np.array([1, 0, 0, 0])
    # now we can run a few iterations, each iteration is one episode
    num_iter = 500
    num_step = 100
    # the same network run 10 times and see if there is any difference in results we get...
    update_config = {'niter': TRAIN_EPOCH * 2, 'lr': 5e-4}
    for itr in range(num_iter):
        print('Iteratiton = %d' % itr)
        best_traj = None
        best_cost = np.inf
        for _ in range(5):
            traj = mppi.run_episode(init_state, num_step)
            # compute the cost...
            costi = np.sum(path_cost(traj.obs[:-1], traj.ctrls[:-1])) + np.sum(term_cost(traj.obs[-1:]))
            print('traj end with ', traj.obs[-1], 'costi = ', costi)
            if costi < best_cost:
                best_cost = costi
                best_traj = traj
        print(best_traj.obs[-1])
        if np.amin(np.linalg.norm(best_traj.obs, np.inf, axis=1)) < 0.2:
            break
        mppi.update_network(mppi.network, best_traj, **update_config)
    


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


def cartpole_dynamics(y, u, g = GRAVITY, m_c = M_C, m_p = M_P, L = LEN, b = BB):
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

def dt_cartpole_dynamics(y,u,dt,g=GRAVITY,m_c=M_C, m_p=M_P, L=LEN,b=BB):
    y[0] += np.pi
    sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m_c, m_p, L, b), (0, dt), y, t_eval = [dt])
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


@memory.cache
def gen_pendulum_trajs(dt, num_trajs, umin, umax):
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


# GAO comment: it seems that using forward euler makes everything much easier. Here I keep this part untouched.
@memory.cache
def gen_cartpole_trajs(dt, num_trajs, umin, umax):
    rng = np.random.default_rng(49)
    trajs = []
    # model = CartPole(choice='Euler')
    for _ in range(num_trajs):
        theta0 = rng.uniform(-0.002, 0.002, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(cartpole, 100)
        for i in range(100):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 1)
            y = dt_cartpole_dynamics(y, u, dt)
            # y = model.pred(y, u)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs


if __name__ == '__main__':
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