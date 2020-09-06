"""
Test the gpytorch version of Gaussian process.
Test batch mode, normal mode, and grad mode.
"""
import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from joblib import Memory

from scipy.integrate import solve_ivp
from autompc.sysid import LargeGaussianProcess

memory = Memory("cache")

cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])

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
    return np.array([omega,
            1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                - (m_c+m_p+m_p)*g*np.sin(theta)
                - b*omega),
            dx,
            1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                (L*omega**2 + g*np.cos(theta)))])

def cartpole_simp_dynamics(y, u, g = 9.8, m = 1, L = 1, b = 0.1):
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
    return np.array([omega,
            g * np.sin(theta)/L - b * omega / (m*L**2) + u * np.cos(theta)/L,
            dx,
            u])

def dt_cartpole_dynamics(y,u,dt,g=9.8,m=1,L=1,b=1.0):
    y = np.copy(y)
    y[0] += np.pi
    sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y = sol.y.reshape((4,))
    #y += dt * cartpole_simp_dynamics(y,u[0],g,m,L,b)
    y[0] -= np.pi
    return y

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
        ctrl_text.set_text("u={:.2f}".format(traj[i,"u"]))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani

dt = 0.01
cartpole.dt = dt

umin = -2.0
umax = 2.0
udmax = 0.25

# Generate trajectories for training
num_trajs = 500

@memory.cache
def gen_trajs(traj_len, num_trajs=num_trajs, dt=dt):
    rng = np.random.default_rng(49)
    trajs = []
    for _ in range(num_trajs):
        theta0 = rng.uniform(-0.002, 0.002, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(cartpole, traj_len)
        for i in range(traj_len):
            traj[i].obs[:] = y
            #if u[0] > umax:
            #    u[0] = umax
            #if u[0] < umin:
            #    u[0] = umin
            #u += rng.uniform(-udmax, udmax, 1)
            u  = rng.uniform(umin, umax, 1)
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

trajs2 = gen_trajs(200)


datasize = 200
num_trajs = 20  # so I have 1000 points....
gp = LargeGaussianProcess(cartpole, niter=40)
mytrajs = [trajs2[i][:datasize] for i in range(num_trajs)]
gp.train(mytrajs)

# test some predictions
state = np.zeros(4)
ctrl = np.random.normal(size=1)
ybase = gp.pred(state, ctrl)
yp, jacx, jacu = gp.pred_diff(state, ctrl)
# batch mode
y_batch = gp.pred_parallel(np.tile(state[None], (10, 1)), np.tile(ctrl[None], (10, 1)))

# test the model by generating one trajectory...
rng = np.random.default_rng(49)
theta0 = rng.uniform(-0.002, 0.002, 1)[0]
y = [theta0, 0.0, 0.0, 0.0]
y2 = y
traj_len = 150
traj = ampc.zeros(cartpole, traj_len)
pred_traj = ampc.zeros(cartpole, traj_len)
for i in range(traj_len):
    traj[i].obs[:] = y
    pred_traj[i].obs[:] = y2
    u  = rng.uniform(umin, umax, 1)
    y = dt_cartpole_dynamics(y, u, dt)
    traj[i].ctrl[:] = u
    pred_traj[i].ctrl[:] = u
    y2 = gp.pred(y, u)
fig, ax = plt.subplots(2, 2)
ax = ax.reshape(-1)
error = pred_traj.obs - traj.obs
for i in range(4):
    ax[i].hist(error[:, i], bins=10)
fig.tight_layout()
plt.show()