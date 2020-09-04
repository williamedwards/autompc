import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from joblib import Memory
from scipy.integrate import solve_ivp


memory = Memory("cache")

pendulum = ampc.System(["ang", "angvel"], ["torque"])
cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
planar_drone = ampc.System(["x", "dx", "y", "dy", "theta", "omega"], ["u1", "u2"])

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

def collect_pendulum_trajs(dt, umin, umax):
    # Generate trajectories for training
    num_trajs = 100
    trajs = gen_pendulum_trajs(dt, num_trajs, umin, umax)
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

def cartpole_dynamics(y, u, g = 9.8, m_c = 1, m_p = 0.1, L = 1, b = 1.00):
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
    return np.array([omega,
            1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                - (m_c+m_p+m_p)*g*np.sin(theta)
                - b*omega),
            dx,
            1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                (L*omega**2 + g*np.cos(theta)))])

def dt_cartpole_dynamics(y,u,dt):
    y = np.copy(np.array(y))
    y[0] += np.pi
    #sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    #if not sol.success:
    #    raise Exception("Integration failed due to {}".format(sol.message))
    #y = sol.y.reshape((4,))
    y = y + dt * cartpole_dynamics(y, u[0])
    y[0] -= np.pi
    return y

def collect_cartpole_trajs(dt, umin, umax, num_trajs=100):
    # Generate trajectories for training
    trajs = gen_cartpole_trajs(dt, num_trajs, umin, umax)
    return trajs


@memory.cache
def gen_cartpole_trajs(dt, num_trajs, umin, umax):
    rng = np.random.default_rng(49)
    trajs = []
    for _ in range(num_trajs):
        theta0 = rng.uniform(-0.002, 0.002, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(cartpole, 200)
        for i in range(200):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 1)
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs


def planar_drone_dynamics(y, u, g, m, r, I):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    x, dx, y2, dy, theta, omega = y
    u1, u2  = u
    return [dx,
            -(u1 + u2) * np.sin(theta),
            dy,
            (u1 + u2) * np.cos(theta) - m * g,
            omega,
            r / I * (u1 - u2)]

def dt_planar_drone_dynamics(y,u,dt,g=0.0,m=1,r=0.25,I=1.0):
    y = np.copy(y)
    sol = solve_ivp(lambda t, y: planar_drone_dynamics(y, u, g, m, r, I), (0, dt), y, 
            t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y = sol.y.reshape((6,))
    #y += dt * np.array(planar_drone_dynamics(y, u, g, m, r, I))
    return y


@memory.cache
def gen_planar_drone_trajs(dt, num_trajs, umin, umax):
    print("Generating planar drone trajs.")
    rng = np.random.default_rng(49)
    trajs = []
    for _ in range(num_trajs):
        theta0 = rng.uniform(-0.02, 0.02, 1)[0]
        x0 = rng.uniform(-0.02, 0.02, 1)[0]
        y0 = rng.uniform(-0.02, 0.02, 1)[0]
        y = [x0, 0.0, y0, 0.0, theta0, 0.0]
        traj = ampc.zeros(planar_drone, 200)
        for i in range(200):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 2)
            y = dt_planar_drone_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs


def collect_planar_drone_trajs(dt, umin, umax):
    # Generate trajectories for training
    num_trajs = 100
    trajs = gen_planar_drone_trajs(dt, num_trajs, umin, umax)
    return trajs


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


def animate_planar_drone(fig, ax, dt, traj, r=0.25):
    ax.grid()

    line, = ax.plot([-1.0, 0.0], [1.0, 0.0], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ctrl_text = ax.text(0.7, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([0.0, 0.0], [0.0, -1.0])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        #i = min(i, ts.shape[0])
        x1 = traj[i, "x"] + r*np.cos(traj[i, "theta"])
        y1 = traj[i, "y"] + r*np.sin(traj[i, "theta"])
        x2 = traj[i, "x"] - r*np.cos(traj[i, "theta"])
        y2 = traj[i, "y"] - r*np.sin(traj[i, "theta"])
        line.set_data([x1, x2], [y1, y2])
        time_text.set_text('t={:.2f}'.format(dt*i))
        ctrl_text.set_text("u1={:.2f}, u2={:.2f}".format(traj[i,"u1"], traj[i,"u2"]))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani
