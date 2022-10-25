# Standard library includes
from os import stat
import sys, time

# External library includes
import numpy as np
import mujoco_py

# Project includes
from ..benchmark import Benchmark
from ...utils.data_generation import *
from ... import System
from ...task import Task
from ...trajectory import Trajectory
from ...costs import Cost, QuadCost


gravity_names = ["HalfCheetahGravityHalf-v0", "HalfCheetahGravityThreeQuarters-v0", 
                 "HalfCheetahGravityOneAndQuarter-v0", "HalfCheetahGravityOneAndHalf-v0"]

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

def gym_dynamics(env, x, u, n_frames=5):
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
    
    for i in range(num_trajs):
        init_obs = env.reset()
        state = env.sim.get_state()
        qpos, qvel = state[1], state[2]
        traj = Trajectory.zeros(system, traj_len)
        
        if len(init_obs) < len(qpos) + len(qvel):
            add_zeros = np.zeros(len(qpos) + len(qvel) - len(init_obs))
            traj[0].obs[:] = np.concatenate([add_zeros, init_obs])
        else:
            traj[0].obs[:] = np.concatenate([qpos, qvel])
                  
        for j in range(1, traj_len):
            action = env.action_space.sample()
            traj[j-1].ctrl[:] = action
            #obs, reward, done, info = env.step(action)
            obs = gym_dynamics(env, traj[j-1].obs[:], action)
            traj[j].obs[:] = obs
        trajs.append(traj)
    return trajs


class GymGravityBenchmark(Benchmark):
    """
    This benchmark is based on the OpenAI Gym Mojoco and gym-extensions. It provides different benchmarks with various
    scales of simulated earth-like gravity, ranging from one half to one and a half of the normal gravity level.
    
    ***GravityHalf-v2: The standard Mujoco OpenAI gym hopper task with gravity scaled by 0.5.
    ***GravityThreeQuarters-v2: The standard Mujoco OpenAI gym hopper task with gravity scaled by 0.75.
    ***GravityOneAndQuarter-v2: The standard Mujoco OpenAI gym hopper task with gravity scaled by 1.25.
    ***GravityOneAndHalf-v2: The standard Mujoco OpenAI gym hopper task with gravity scaled by 1.5.
    """
    def __init__(self, name = "HalfCheetah-v2", data_gen_method="uniform_random", gravity_rate=1.0):
        # import gym, mujoco_py
        from gym_extensions.continuous import mujoco
        import gym
        print(name)
        
        env = gym.make(name)
        self.env = env
        
        # Modify the gravity 
        gravity = [0., 0., -9.81*gravity_rate]
        env.model.opt.gravity[:] = gravity
        print('gravity', env.model.opt.gravity)
        
        state = env.sim.get_state()
        qpos = state[1]
        qvel = state[2]

        x_num = len(qpos) + len(qvel)
        u_num = env.action_space.shape[0]
        system = ampc.System([f"x{i}" for i in range(x_num)], [f"u{i}" for i in range(u_num)], env.dt)

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
        return gym_dynamics(self.env,x,u)

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

