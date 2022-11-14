# Created by William Edwards (wre2@illinois.edu), 2021-01-09

# Standard library includes
from os import stat
import sys, time

# External library includes
import numpy as np
# import mujoco_py
import mujoco_py

# Project includes
from .control_benchmark import ControlBenchmark
from ..utils.data_generation import *
from .. import System
from ..task import Task
from ..trajectory import Trajectory
from ..costs import Cost, QuadCost
from ..ocp import QuadCostTransformer

def viz_halfcheetah_traj(env, traj, repeat):
    for _ in range(repeat):
        env.reset()
        qpos = traj[0].obs[:9]
        qvel = traj[0].obs[9:]
        env.set_state(qpos, qvel)
        for i in range(len(traj)):
            u = traj[i].ctrl
            env.step(u)
            env.render()
            time.sleep(0.05)
        time.sleep(1)

def halfcheetah_dynamics(env, x, u, n_frames=5):
    old_state = env.sim.get_state()
    old_qpos = old_state[1]
    old_qvel = old_state[2]

    qpos = x[:len(old_qpos)]
    qvel = x[len(old_qpos):len(old_qpos)+len(old_qvel)]

    # Represents a snapshot of the simulator's state.
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
    env.sim.set_state(new_state)
    #env.sim.forward()

    env.sim.data.ctrl[:] = u
    for _ in range(n_frames):
        env.sim.step()
    
    new_qpos = env.sim.data.qpos
    new_qvel = env.sim.data.qvel
    out = np.concatenate([new_qpos, new_qvel])

    return out

# class HalfcheetahCost(Cost):
#     def __init__(self, env):
#         Cost.__init__(self,None)
#         self.env = env

#     def __call__(self, traj):
#         cum_reward = 0.0
#         for i in range(len(traj)-1):
#             reward_ctrl = -0.1 * np.square(traj[i].ctrl).sum()
#             reward_run = (traj[i+1, "x0"] - traj[i, "x0"]) / self.env.dt
#             cum_reward += reward_ctrl + reward_run
#         return 200 - cum_reward

#     def incremental(self,obs,ctrl):
#         raise NotImplementedError

#     def terminal(self,obs):
#         raise NotImplementedError


def gen_trajs(env, system, num_trajs=1000, traj_len=1000, seed=42):
    rng = np.random.default_rng(seed)
    trajs = []
    env.seed(int(rng.integers(1 << 30)))
    env.action_space.seed(int(rng.integers(1 << 30)))
    state = env.sim.get_state()
    qpos, qvel = state[1], state[2]
    for i in range(num_trajs):
        init_obs = env.reset()
        traj = Trajectory.zeros(system, traj_len)

        add_zeros = np.zeros(len(qpos) + len(qvel) - len(init_obs))
        traj[0].obs[:] = np.concatenate([add_zeros, init_obs])       

        for j in range(1, traj_len):
            action = env.action_space.sample()
            traj[j-1].ctrl[:] = action
            #obs, reward, done, info = env.step(action)
            obs = halfcheetah_dynamics(env, traj[j-1].obs[:], action)
            traj[j].obs[:] = obs
        trajs.append(traj)
    return trajs


class HalfcheetahBenchmark(ControlBenchmark):
    """
    This benchmark uses the OpenAI gym halfcheetah benchmark and is consistent with the
    experiments in the ICRA 2021 paper. The benchmark reuqires OpenAI gym and mujoco_py
    to be installed.  The performance metric is
    :math:`200-R` where :math:`R` is the gym reward.
    """

    def __init__(self, data_gen_method="uniform_random"):
        name = "halfcheetah"

        import gym, mujoco_py
        print(name)

        env = gym.make(name)
        self.env = env
        state = env.sim.get_state()
        qpos = state[1]
        qvel = state[2]

        x_num = len(qpos) + len(qvel)
        u_num = env.action_space.shape[0]
        system = ampc.System([f"x{i}" for i in range(x_num)], [f"u{i}" for i in range(u_num)], env.dt)

        system = ampc.System([f"x{i}" for i in range(18)], [f"u{i}" for i in range(6)], env.dt)

        system.dt = env.dt

        cost = HalfcheetahCost(env)
        task = Task(system,cost)
        task.set_ctrl_bounds(env.action_space.low, env.action_space.high)
        init_obs = np.concatenate([env.init_qpos, env.init_qvel])
        task.set_init_obs(init_obs)
        task.set_num_steps(200)

        transformer = QuadCostTransformer(system)
        for obs in system.observations:
            if not obs in ["x1", "x6", "x7", "x8", "x9"]:
                transformer.fix_Q_value(obs, 0.0)
            if not obs in ["x1", "x9"]:
                transformer.fix_F_value(obs, 0.0)
        transformer.set_tunable_goal("x9", lower_bound=0.0, upper_bound=5.0, default=1.0)
        self.ocp_transformer = transformer

        super().__init__(name, system, task, data_gen_method)

    def dynamics(self, x, u):
        return halfcheetah_dynamics(self.env,x,u)

    def gen_trajs(self, seed, n_trajs, traj_len=200):
        return gen_trajs(self.env, self.system, n_trajs, traj_len, seed)

    def visualize(self, traj, repeat):
        """
        Visualize the half-cheetah trajectory using Gym functions.

        Parameters
        ----------
        traj : Trajectory
            Trajectory to visualize

        repeat : int
            Number of times to repeat trajectory in visualization
        """
        viz_halfcheetah_traj(self.env, traj, repeat)

    @staticmethod
    def data_gen_methods():
        return ["uniform_random"]

"""
Benchmark for Meta-world
"""
def meta_dynamics(env, x, u, n_frames=5):
    """
    x: traj[j-1].obs[:]
    u: action
    """
    old_state = env.sim.get_state()
    old_qpos = old_state[1]
    old_qvel = old_state[2]

    qpos = x[:len(old_qpos)]
    qvel = x[len(old_qpos):len(old_qpos)+len(old_qvel)]

    # Represents a snapshot of the simulator's state.
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
    env.sim.set_state(new_state)
    #env.sim.forward()
    
    # print("u", u)
    # # print("new state", new_state)
    # print("env control", env.sim.data.ctrl[:]) #sim.data.ctrl is the control signal that is sent to the actuators in the simulation.
    # env.sim.data.ctrl[:] = u
    for _ in range(n_frames):
        env.sim.step()
    
    new_qpos = env.sim.data.qpos
    new_qvel = env.sim.data.qvel
    out = np.concatenate([new_qpos, new_qvel])
    if len(out) < len(x):
        add_zeros = np.zeros(len(x)-len(out))
        out = np.concatenate([out, add_zeros])

    return out

def meta_gen_trajs(env, system, num_trajs=1000, traj_len=1000, seed=42):
    rng = np.random.default_rng(seed)
    trajs = []
    env.seed(int(rng.integers(1 << 30)))
    env.action_space.seed(int(rng.integers(1 << 30)))
    state = env.sim.get_state()
    qpos, qvel = state[1], state[2]
    for i in range(num_trajs):
        init_obs = env.reset()
        # print("observation len", len(init_obs))
        traj = Trajectory.zeros(system, traj_len)

        if len(init_obs) < len(qpos) + len(qvel):
            add_zeros = np.zeros(len(qpos) + len(qvel) - len(init_obs))
            traj[0].obs[:] = np.concatenate([add_zeros, init_obs])
        else:
            traj[0].obs[:] = init_obs       

        for j in range(1, traj_len):
            action = env.action_space.sample()
            traj[j-1].ctrl[:] = action
            #obs, reward, done, info = env.step(action)
            obs = meta_dynamics(env, traj[j-1].obs[:], action)
            traj[j].obs[:] = obs
        trajs.append(traj)
    return trajs


class MetaBenchmark(Benchmark):
    """
    The performance metric is
    :math:`200-R` where :math:`R` is the gym reward.
    """
    def __init__(self, name = "HalfCheetah-v2", data_gen_method="uniform_random"):
        import metaworld
        import random

        names = ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 
                'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 
                'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 
                'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 
                'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 
                'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 
                'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 
                'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 
                'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 
                'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 
                'sweep-v2', 'window-open-v2', 'window-close-v2']

        # name  = random.choice(names)
        # name = names[2]
        print(name)
        self.name = name

        ml1 = metaworld.ML1(name)
        env = ml1.train_classes[name]()
        self.env = env

        # random.seed(2022)
        # task = random.choice(ml1.train_tasks)
        task = ml1.train_tasks[0]
        env.set_task(task)

        state = env.sim.get_state()
        qpos = state[1]
        qvel = state[2]
        print('qpos', qpos)
        print('qvel', qvel)
        obs_shape = env.observation_space.shape[0]
        print('obs', env._get_obs())
        # exit()
        print('qpos {}, qvel {}'.format(len(qpos), len(qvel)))
        print('obs shape {}, state shape {} \n'.format(obs_shape, env.action_space.shape[0]))

        if obs_shape < len(qpos) + len(qvel):
            x_num = len(qpos) + len(qvel)
        else:
            x_num = obs_shape

        u_num = env.action_space.shape[0]
        # print(x_num, u_num)
        system = ampc.System([f"x{i}" for i in range(x_num)], [f"u{i}" for i in range(u_num)], env.dt) #18, 6

        system.dt = env.dt
        task = Task(system)

        # cost = HalfcheetahCost(env)
        # task = Task(system,cost)
        # task.set_ctrl_bounds(env.action_space.low, env.action_space.high)
        # init_obs = np.concatenate([env.init_qpos, env.init_qvel])
        # task.set_init_obs(init_obs)
        # task.set_num_steps(200)

        # factory = QuadCost(system, goal = np.zeros(system.obs_dim))
        # for obs in system.observations:
        #     if not obs in ["x1", "x6", "x7", "x8", "x9"]:
        #         factory.fix_Q_value(obs, 0.0)
        #     if not obs in ["x1", "x9"]:
        #         factory.fix_F_value(obs, 0.0)
        # factory.set_tunable_goal("x9", lower_bound=0.0, upper_bound=5.0, default=1.0)
        # self.cost_factory = factory


        super().__init__(name, system, task, data_gen_method)

    def dynamics(self, x, u):
        return meta_dynamics(self.env,x,u)

    def gen_trajs(self, seed, n_trajs, traj_len=200):
        return meta_gen_trajs(self.env, self.system, n_trajs, traj_len, seed)

    def visualize(self, traj, repeat):
        """
        Visualize the half-cheetah trajectory using Gym functions.

        Parameters
        ----------
        traj : Trajectory
            Trajectory to visualize

        repeat : int
            Number of times to repeat trajectory in visualization
        """
        viz_halfcheetah_traj(self.env, traj, repeat)

    @staticmethod
    def data_gen_methods():
        return ["uniform_random"]
