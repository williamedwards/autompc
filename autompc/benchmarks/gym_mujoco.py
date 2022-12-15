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

gym_names = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Swimmer-v2", "InvertedPendulum-v2", 
              "Reacher-v2", "Pusher-v2", "InvertedDoublePendulum-v2", 
              "Ant-v2", "Humanoid-v2", "HumanoidStandup-v2"]

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
    old_state = env.sim.get_state()
    old_qpos = old_state[1]
    old_qvel = old_state[2]

    qpos = x[:len(old_qpos)]
    qvel = x[len(old_qpos):len(old_qpos)+len(old_qvel)]

    # Represents a snapshot of the simulator's state.
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
    env.sim.set_state(new_state)

    env.sim.data.ctrl[:] = u
    for _ in range(n_frames):
        env.sim.step()
    
    new_qpos = env.sim.data.qpos
    new_qvel = env.sim.data.qvel
    out = np.concatenate([new_qpos, new_qvel])

    return out

def gym_reward(env, x, u, n_frames=5):
    old_state = env.sim.get_state()
    old_qpos = old_state[1]
    old_qvel = old_state[2]

    qpos = x[:len(old_qpos)]
    qvel = x[len(old_qpos):len(old_qpos)+len(old_qvel)]

    # Represents a snapshot of the simulator's state.
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
    env.sim.set_state(new_state)

    total_reward = 0
    # env.sim.data.ctrl[:] = u
    _, total_reward, _, _  = env.step(u)
    # for _ in range(n_frames):
    #     _, reward, _, _, _ = env.sim.step()
    #     total_reward += reward

    return total_reward

class GymRewardCost(Cost):
    def __init__(self, env, cost_offset=200):
        Cost.__init__(self,None)
        self.env = env
        self._cost_offset = cost_offset

    def __call__(self, traj, cost_offset=200):
        cum_reward = 0.0
        for i in range(len(traj)-1):
            cum_reward += gym_reward(self.env, traj[i].obs, traj[i].ctrl)
        return self._cost_offset - cum_reward

    def incremental(self,obs,ctrl):
        raise NotImplementedError

    def terminal(self,obs):
        raise NotImplementedError


def _get_init_obs(env):
    env.reset()
    qpos = env.sim.data.qpos
    qvel = env.sim.data.qvel
    return np.concatenate([qpos, qvel])

def gen_trajs(env, system, num_trajs=1000, traj_len=1000, seed=42):
    rng = np.random.default_rng(seed)
    trajs = []
    env.seed(int(rng.integers(1 << 30)))
    env.action_space.seed(int(rng.integers(1 << 30)))
    
    for i in range(num_trajs):
        init_obs = _get_init_obs(env)
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
            obs = gym_dynamics(env, traj[j-1].obs[:], action)
            traj[j].obs[:] = obs
        trajs.append(traj)
    return trajs


class GymMujocoBenchmark(ControlBenchmark):
    """
    This benchmark uses the OpenAI gym halfcheetah benchmark and is consistent with the
    experiments in the ICRA 2021 paper. The benchmark reuqires OpenAI gym and mujoco_py
    to be installed.  The performance metric is
    :math:`200-R` where :math:`R` is the gym reward.
    """
    def __init__(self, name = "HalfCheetah-v2", data_gen_method="uniform_random"):
        import gym, mujoco_py

        env = gym.make(name)
        env.seed(0)
        self.env = env
        state = env.sim.get_state()
        qpos = state[1]
        qvel = state[2]

        x_num = len(qpos) + len(qvel)
        u_num = env.action_space.shape[0]
        system = ampc.System([f"x{i}" for i in range(x_num)], [f"u{i}" for i in range(u_num)], env.dt)

        system.dt = env.dt
        task = Task(system)
        task.set_cost(GymRewardCost(env))
        task.set_init_obs(_get_init_obs(env))

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