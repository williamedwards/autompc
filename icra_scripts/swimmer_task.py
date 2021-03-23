# Standard library includes
import sys, time

# External library includes
import numpy as np
import gym
import mujoco_py
import autompc as ampc
from autompc.tasks import Task, QuadCost

# Project includes
from data_generation import *

swimmer = ampc.System([f"x{i}" for i in range(10)], [f"u{i}" for i in range(2)])
env = gym.make("Swimmer-v2")
swimmer.dt = env.dt

from collections import namedtuple
from joblib import Memory
from benchmark import Benchmark

memory = Memory("cache")

TaskInfo = namedtuple("TaskInfo", ["name", "system", "task", "init_obs", 
    "dynamics", "perf_metric", "gen_sysid_trajs", "gen_surr_trajs"])

def viz_swimmer_traj(traj, repeat):
    for _ in range(repeat):
        env.reset()
        qpos = traj[0].obs[:5]
        qvel = traj[0].obs[5:]
        env.set_state(qpos, qvel)
        for i in range(len(traj)):
            u = traj[i].ctrl
            env.step(u)
            env.render()
            time.sleep(0.05)
        time.sleep(1)

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

def swimmer_task():
    benchmark = SwimmerBenchmark("uniform_random")
    return icra_task_from_benchmark(benchmark)

def swimmer_dynamics(x, u, n_frames=5):
    old_state = env.sim.get_state()
    old_qpos = old_state[1]
    qpos = x[:len(old_qpos)]
    qvel = x[len(old_qpos):]
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
            old_state.act, old_state.udd_state)
    env.sim.set_state(new_state)
    #env.sim.forward()
    env.sim.data.ctrl[:] = u
    for _ in range(n_frames):
        env.sim.step()
    new_qpos = env.sim.data.qpos
    new_qvel = env.sim.data.qvel

    return np.concatenate([new_qpos, new_qvel])

class SwimmerBenchmark(Benchmark):
    def __init__(self, data_gen_method):
        name = "swimmer"
        system = swimmer

        #Q = np.zeros((system.obs_dim, system.obs_dim))
        #Q[5,5] = 1.0
        #R = 0.01 * np.eye(system.ctrl_dim)
        #F = np.eye(system.obs_dim)
        #cost = QuadCost(system, Q, R, F, x0=np.array([0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,
        #    0.0,0.0]))
        Q = np.eye(system.obs_dim)
        R = np.eye(system.ctrl_dim)
        F = np.eye(system.obs_dim)
        cost = QuadCost(system, Q, R, F)
        task = Task(system)
        task.set_cost(cost)
        task.set_ctrl_bounds(env.action_space.low, env.action_space.high)
        init_obs = np.concatenate([env.init_qpos, env.init_qvel])


        super().__init__(name, system, task, init_obs, data_gen_method) 

    def viz_traj(self, traj, repeat=1):
        viz_halfcheetah_traj(traj, repeat)

    def perf_metric(self, traj):
        cum_reward = 0.0
        for i in range(len(traj)-1):
            reward_ctrl = -0.0001 * np.square(traj[i].ctrl).sum()
            reward_run = (traj[i+1, "x0"] - traj[i, "x0"]) / env.dt
            cum_reward += reward_ctrl + reward_run
        return 200 - cum_reward

    def dynamics(self, x, u):
        return swimmer_dynamics(x,u)

    def _gen_trajs(self, n_trajs, rng):
        init_min = self.init_obs
        init_max = self.init_obs
        traj_len = 200
        if self._data_gen_method == "uniform_random":
            return uniform_random_generate(self.system, self.task, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max,
                    traj_len=traj_len, n_trajs=n_trajs)
        elif self._data_gen_method == "periodic_control":
            return periodic_control_generate(self.system, self.task, self.dynamics, rng, 
                    init_min=init_min, init_max=init_max, 
                    U_1=np.ones(self.system.ctrl_dim), traj_len=traj_len, n_trajs=n_trajs)
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
