# Standard library includes
from os import stat
import sys, time

# External library includes
import numpy as np
# import mujoco_py
import mujoco_py

# Project includes
from ..benchmark import Benchmark
from ...utils.data_generation import *
from ... import System
from ...task import Task
from ...trajectory import Trajectory
from ...costs import Cost, QuadCost

gym_names = ["HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Swimmer-v4", "InvertedPendulum-v4", 
              "Reacher-v4", "Pusher-v4", "InvertedDoublePendulum-v4", 
              "Ant-v4", "Humanoid-v4", "HumanoidStandup-v4"]

def viz_gym_traj(env, traj, repeat):
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
    old_state = env.get_state()
    old_qpos = old_state[1]
    old_qvel = old_state[2]

    qpos = x[:len(old_qpos)]
    qvel = x[len(old_qpos):len(old_qpos)+len(old_qvel)]

    # Represents a snapshot of the simulator's state.
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
    env.set_state(new_state)

    env.data.ctrl[:] = u
    for _ in range(n_frames):
        env.step()
    
    new_qpos = env.data.qpos
    new_qvel = env.data.qvel
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
        state = env.get_state()
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


class GymBenchmark(Benchmark):
    """
    This benchmark uses the OpenAI gym halfcheetah benchmark and is consistent with the
    experiments in the ICRA 2021 paper. The benchmark reuqires OpenAI gym and mujoco_py
    to be installed.  The performance metric is
    :math:`200-R` where :math:`R` is the gym reward.
    """
    def __init__(self, name = "HalfCheetah-v4", data_gen_method="uniform_random"):
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
        viz_gym_traj(self.env, traj, repeat)

    @staticmethod
    def data_gen_methods():
        return ["uniform_random"]

