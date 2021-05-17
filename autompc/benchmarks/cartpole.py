# Created by William Edwards (wre2@illinois.edu), 2021-01-09

# Standard library includes
import sys

# External library includes
import numpy as np
from .. import System
from ..tasks import Task
from ..costs import ThresholdCost

# Project includes
from .benchmark import Benchmark
from ..utils.data_generation import *

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
    y += dt * cartpole_simp_dynamics(y,u[0],g,m,L,b)
    return y


class CartpoleSwingupBenchmark(Benchmark):
    def __init__(self, data_gen_method="uniform_random"):
        name = "cartpole_swingup"
        system = ampc.System(["theta", "omega", "x", "dx"], ["u"])
        system.dt = 0.05

        cost = ThresholdCost(system, goal=np.zeros(4), threshold=0.2, obs_range=(0,2))
        task = Task(system)
        task.set_cost(cost)
        task.set_ctrl_bound("u", -20.0, 20.0)
        init_obs = np.array([3.1, 0.0, 0.0, 0.0])
        task.set_init_obs(init_obs)
        task.set_num_steps(200)

        super().__init__(name, system, task, data_gen_method)

    def perf_metric(self, traj):
        return threshold_metric(goal=np.zeros(2), threshold=0.2, obs_range=(0,2),
                traj=traj)

    def dynamics(self, x, u):
        return dt_cartpole_dynamics(x,u,self.system.dt,g=0.8,m=1,L=1,b=1.0)

    def _gen_trajs(self, n_trajs, traj_len, rng):
        init_min = np.array([-1.0, 0.0, 0.0, 0.0])
        init_max = np.array([1.0, 0.0, 0.0, 0.0])
        if self._data_gen_method == "uniform_random":
            return uniform_random_generate(self.system, self.task, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max,
                    traj_len=traj_len, n_trajs=n_trajs)
        elif self._data_gen_method == "periodic_control":
            return periodic_control_generate(self.system, self.task, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max, U_1=np.ones(1),
                    traj_len=traj_len, n_trajs=n_trajs)
        elif self._data_gen_method == "multisine":
            return multisine_generate(self.system, self.task, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max, n_freqs=20,
                    traj_len=traj_len, n_trajs=n_trajs)
        elif self._data_gen_method == "random_walk":
            return random_walk_generate(self.system, self.task, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max, walk_rate=1.0,
                    traj_len=traj_len, n_trajs=n_trajs)

    def gen_trajs(self, seed, n_trajs, traj_len=200):
        rng = np.random.default_rng(seed)
        return self._gen_trajs(n_trajs, traj_len, rng)


    @staticmethod
    def data_gen_methods():
        return ["uniform_random", "periodic_control", "multisine", "random_walk"]
