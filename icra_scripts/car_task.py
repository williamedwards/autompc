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

def car_task():
    benchmark = CarBenchmark("uniform_random")
    return icra_task_from_benchmark(benchmark)

def car_dynamics(y, u, L=1):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    x, y, theta, speed = y
    u_steer, u_accel = u
    return np.array([speed * np.cos(theta),
            speed * np.sin(theta),
            speed / L * np.tan(u_steer),
            u_accel])

def dt_car_dynamics(y,u,dt,L=1.0):
    return y + dt * car_dynamics(y,u,L)

class CarBenchmark(Benchmark):
    def __init__(self, data_gen_method):
        name = "car"
        system = ampc.System(["x", "y", "theta", "speed"], ["u_steer", "u_accel"])
        system.dt = 0.05

        Q = np.eye(4)
        R = np.eye(2)
        F = np.eye(4)
        cost = QuadCost(system, Q, R, F)
        task = Task(system)
        task.set_cost(cost)
        task.set_ctrl_bound("u_accel", -1.0, 1.0)
        task.set_ctrl_bound("u_steer", -1.5, 1.5)
        init_obs = np.array([0.0, -5.0, 3.1, 0.0])

        super().__init__(name, system, task, init_obs, data_gen_method) 

    def perf_metric(self, traj):
        return threshold_metric(goal=np.zeros(3), threshold=0.5, obs_range=(0,3),
                traj=traj)

    def dynamics(self, x, u):
        return dt_car_dynamics(x,u,self.system.dt,L=1)

    def _gen_trajs(self, n_trajs, rng):
        init_min = np.array([-1.0, -1.0, -1.0, -1.0])
        init_max = np.array([1.0, 1.0, 1.0, 1.0])
        traj_len = 200
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
