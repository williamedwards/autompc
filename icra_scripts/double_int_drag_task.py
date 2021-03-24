# Standard library includes
import sys, time

# External library includes
import numpy as np
import gym
import mujoco_py
import autompc as ampc
from autompc.tasks import Task, QuadCost

# Project includes
from perf_metrics import threshold_metric
from data_generation import *


from collections import namedtuple
from joblib import Memory
from benchmark import Benchmark

memory = Memory("cache")

TaskInfo = namedtuple("TaskInfo", ["name", "system", "task", "init_obs", 
    "dynamics", "perf_metric", "gen_sysid_trajs", "gen_surr_trajs"])



@memory.cache
def gen_trajs(benchmark, num_trajs, seed):
    rng = np.random.default_rng(seed)
    return benchmark._gen_trajs(num_trajs, rng)

def icra_task_from_benchmark(benchmark):
    gen_sysid_trajs = lambda seed, n_trajs=500: gen_trajs(benchmark,
            n_trajs, seed=seed)
    gen_surr_trajs = lambda seed, n_trajs=500: gen_trajs(benchmark,
            n_trajs, seed=seed)
    return TaskInfo(name=benchmark.name,
            system=benchmark.system,
            task=benchmark.task,
            init_obs=benchmark.init_obs,
            dynamics=benchmark.dynamics,
            perf_metric=benchmark.perf_metric,
            gen_sysid_trajs=gen_sysid_trajs,
            gen_surr_trajs=gen_surr_trajs)

def double_int_drag_task():
    benchmark = DoubleIntDragBenchmark("uniform_random")
    return icra_task_from_benchmark(benchmark)

def doubleint_drag_dynamics(y, u, drag=0.1):
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
    return np.array([dx, u - drag*dx**2])

def dt_doubleint_drag_dynamics(y,u,dt):
    y += dt * doubleint_drag_dynamics(y,u[0])
    return y

class DoubleIntDragBenchmark(Benchmark):
    def __init__(self, data_gen_method):
        name = "double_int_drag"
        system = ampc.System(["x", "dx"], ["u"])
        system.dt = 0.05

        Q = np.eye(2)
        R = np.eye(1)
        F = np.eye(2)
        cost = QuadCost(system, Q, R, F)
        task = Task(system)
        task.set_cost(cost)
        task.set_ctrl_bound("u", -1.0, 1.0)
        init_obs = np.array([10.0, 0.0])

        super().__init__(name, system, task, init_obs, data_gen_method) 

    def perf_metric(self, traj):
        return threshold_metric(goal=np.zeros(2), threshold=0.2, obs_range=(0,2),
                traj=traj)

    def dynamics(self, x, u):
        return dt_doubleint_drag_dynamics(x,u,self.system.dt)

    def _gen_trajs(self, n_trajs, rng):
        if self._data_gen_method == "uniform_random":
            return uniform_random_generate(self.system, self.task, self.dynamics, rng, 
                    init_min=[10.0, 0.0], init_max=[10.0, 0.0],
                    traj_len=200, n_trajs=n_trajs)
        elif self._data_gen_method == "periodic_control":
            return periodic_control_generate(self.system, self.task, self.dynamics, rng, 
                    init_min=[10.0, 0.0], init_max=[10.0, 0.0], U_1=np.ones(1),
                    traj_len=200, n_trajs=n_trajs)
        elif self._data_gen_method == "multisine":
            return multisine_generate(self.system, self.task, self.dynamics, rng, 
                    init_min=[10.0, 0.0], init_max=[10.0, 0.0], n_freqs=20,
                    traj_len=200, n_trajs=n_trajs)
        elif self._data_gen_method == "random_walk":
            return random_walk_generate(self.system, self.task, self.dynamics, rng, 
                    init_min=[10.0, 0.0], init_max=[10.0, 0.0], walk_rate=1.0,
                    traj_len=200, n_trajs=n_trajs)

    def gen_sysid_trajs(self, n_trajs, seed):
        rng = np.random.default_rng(seed)
        return self._gen_trajs(n_trajs, rng)

    def gen_surr_trajs(self, n_trajs, seed):
        rng = np.random.default_rng(seed)
        rng = np.random.default_rng(rng.integers(1<<30))
        return self._gen_trajs(n_trajs, rng)

    @staticmethod
    def data_gen_methods():
        return ["uniform_random", "periodic_control", "multisine", "random_walk"]

