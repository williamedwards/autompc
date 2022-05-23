# Created by William Edwards (wre2@illinois.edu), 2021-01-09

# Standard library includes
import sys

# External library includes
import numpy as np
import matplotlib.animation as animation

# Project includes
from .benchmark import Benchmark
from ..utils.data_generation import *
from .. import System
from ..task import Task
from ..costs import ThresholdCost

def cartpole_simp_dynamics(y, u, g = 9.8, m = 1, L = 1, b = 0.1, m_pole=0.0):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart pole
    """
    theta, omega, x, dx = y

    #From OpenAI gym
    # costheta = np.cos(theta)
    # sintheta = np.sin(theta)
    # # For the interested reader:
    # # https://coneural.org/florian/papers/05_cart_pole.pdf
    # temp = (u + m_pole * L*0.5 * omega**2 * sintheta) / (m+m_pole)
    # thetaacc = (g * sintheta - costheta * temp) / (
    #     L*0.5 * (4.0 / 3.0 - m_pole * costheta**2 / (m+m_pole)))
    # xacc = temp - m_pole*L*0.5 * thetaacc * costheta / (m+m_pole)
    # return np.array([omega,thetaacc,dx,xacc])
    return np.array([omega,
            g * np.sin(theta)/L - b * omega / (m*L**2) + u * np.cos(theta)/L,
            dx,
            u])

def dt_cartpole_dynamics(y,u,dt,g=9.8,m=1,L=1,b=1):
    return y + dt * cartpole_simp_dynamics(y,u[0],g,m,L,b)

class CartpoleSwingupBenchmark(Benchmark):
    """
    This benchmark uses the cartpole system and is consistent with the
    experiments in the ICRA 2021 paper. The task is to move the pole
    from the down position to the up position. The performance metric
    returns 1 for every observation which is more than 0.2 away from the goal
    in either the angle or angular velocity dimensions, and 0 otherwise.
    """
    def __init__(self, data_gen_method="uniform_random"):
        name = "cartpole_swingup"
        system = ampc.System(["theta", "omega", "x", "dx"], ["u"])
        system.dt = 0.05

        cost = ThresholdCost(system, goal=np.zeros(4), threshold=0.2, obs_range=(0,3))
        task = Task(system,cost)
        task.set_ctrl_bound("u", -20.0, 20.0)
        init_obs = np.array([3.1, 0.0, 0.0, 0.0])
        task.set_init_obs(init_obs)
        task.set_num_steps(200)

        super().__init__(name, system, task, data_gen_method)
    
    def dynamics(self,x,u):
        return dt_cartpole_dynamics(x,u,self.system.dt,g=9.8,m=1,L=1,b=1.0)

    
    def visualize(self, fig, ax, traj, margin=5.0):
        """
        Visualize the cartpole trajectory.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to generate visualization in.

        ax : matplotlib.axes.Axes
            Axes to create visualization in.

        traj : Trajectory
            Trajectory to visualize

        margin : float
            Shift the viewing window by this amount when the 
            cartpole reaches the edge of the screen
        """
        ax.plot([-10000, 10000.0], [0.0, 0.0], "k-", lw=1)
        ax.set_xlim([-10.0, 10.0])
        ax.set_ylim([-2.0, 2.0])
        ax.set_aspect("equal")
        dt = self.system.dt

        line, = ax.plot([0.0, 0.0], [0.0, -1.0], 'o-', lw=2)
        time_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        ctrl_text = ax.text(0.7, 0.85, '', transform=ax.transAxes)

        def init():
            line.set_data([0.0, 0.0], [0.0, -1.0])
            time_text.set_text('')
            return line, time_text

        nframes = len(traj)
        def animate(i):
            i %= nframes
            i = min(i, len(traj)-1)
            if i == 0:
                ax.set_xlim([-10.0, 10.0])
            #i = min(i, ts.shape[0])
            line.set_data([traj[i,"x"], traj[i,"x"]+np.sin(traj[i,"theta"]+np.pi)], 
                    [0, -np.cos(traj[i,"theta"] + np.pi)])
            time_text.set_text('t={:.2f}'.format(dt*i))
            ctrl_text.set_text("u={:.2f}".format(traj[i,"u"]))
            xmin, xmax = ax.get_xlim()
            if traj[i, "x"] < xmin:
                ax.set_xlim([traj[i,"x"] - margin, traj[i,"x"] + 20.0 - margin])
            if traj[i, "x"] > xmax:
                ax.set_xlim([traj[i,"x"] - 20.0 + margin, traj[i,"x"] + margin])
            return line, time_text

        anim = animation.FuncAnimation(fig, animate, frames=nframes, interval=dt*1000.0,
                blit=False, init_func=init)

        return anim

    def _gen_trajs(self, n_trajs, traj_len, rng):
        init_min = np.array([-1.0, 0.0, 0.0, 0.0])
        init_max = np.array([1.0, 0.0, 0.0, 0.0])
        ocp = self.task.get_ocp()
        if self._data_gen_method == "uniform_random":
            return uniform_random_generate(self.system, ocp, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max,
                    traj_len=traj_len, n_trajs=n_trajs)
        elif self._data_gen_method == "periodic_control":
            return periodic_control_generate(self.system, ocp, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max, U_1=np.ones(1),
                    traj_len=traj_len, n_trajs=n_trajs)
        elif self._data_gen_method == "multisine":
            return multisine_generate(self.system, ocp, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max, n_freqs=20,
                    traj_len=traj_len, n_trajs=n_trajs)
        elif self._data_gen_method == "random_walk":
            return random_walk_generate(self.system, ocp, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max, walk_rate=40.0,
                    traj_len=traj_len, n_trajs=n_trajs)

    def gen_trajs(self, seed, n_trajs, traj_len=200):
        rng = np.random.default_rng(seed)
        return self._gen_trajs(n_trajs, traj_len, rng)


    @staticmethod
    def data_gen_methods():
        return ["uniform_random", "periodic_control", "multisine", "random_walk"]
