# Created by William Edwards (wre2@illinois.edu), 2021-01-09

# Standard library includes
import sys

# External library includes
import numpy as np
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# Project includes
from .benchmark import Benchmark
from ..utils.data_generation import *
from .. import System
from ..task import Task
from ..costs import ThresholdCost

def doubleint_dynamics(y, u):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    x, dx = y
    return np.array([dx, u])

def dt_doubleint_dynamics(y,u,dt):
    y += dt * doubleint_dynamics(y,u[0])
    return y

class DoubleIntegratorBenchmark(Benchmark):
    def __init__(self, data_gen_method="uniform_random"):
        name = "double_integrator"
        system = ampc.System(["x", "dx"], ["u"])
        system.dt = 0.05

        cost = ThresholdCost(system, goal=np.zeros(2), threshold=0.1, 
                obs_range=(0,2))
        task = Task(system,cost)
        task.set_ctrl_bound("u", -10.0, 10.0)
        init_obs = np.array([10.0, 0.0])
        task.set_init_obs(init_obs)
        task.set_num_steps(200)

        super().__init__(name, system, task,  data_gen_method) 

    def dynamics(self, x, u):
        return dt_doubleint_dynamics(x,u,self.system.dt)

    def _gen_trajs(self, n_trajs, traj_len, rng):
        init_min = np.array([-10.0, -5.0])
        init_max = np.array([10.0, 5.0])
        ocp = self.task.get_ocp()
        if self._data_gen_method == "uniform_random":
            return uniform_random_generate(self.system, ocp, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max, 
                    traj_len=traj_len, n_trajs=n_trajs)
        elif self._data_gen_method == "periodic_control":
            return periodic_control_generate(self.system, ocp, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max, 
                    traj_len=traj_len, n_trajs=n_trajs, U_1=np.ones(self.system.ctrl_dim),
                    umin=-2.0, umax=2.0)
        elif self._data_gen_method == "multisine":
            return multisine_generate(self.system, ocp, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max, 
                    traj_len=traj_len, n_trajs=n_trajs, n_freqs=20,
                    umin=-2.0, umax=2.0)
        elif self._data_gen_method == "random_walk":
            return random_walk_generate(self.system, ocp, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max, 
                    traj_len=traj_len, n_trajs=n_trajs, walk_rate=1.0,
                    umin=-2.0, umax=2.0)

    def gen_trajs(self, seed, n_trajs, traj_len=200):
        rng = np.random.default_rng(seed)
        return self._gen_trajs(n_trajs, traj_len, rng)

    @staticmethod
    def data_gen_methods():
        return ["uniform_random", "periodic_control", "multisine",
                "random_walk"]

