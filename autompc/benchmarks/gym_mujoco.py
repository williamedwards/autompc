# Standard library includes
from os import stat
import sys, time
from pathlib import Path
import subprocess

# External library includes
import numpy as np
# import mujoco_py
import mujoco_py
from PIL import Image

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

def viz_gym_traj(env, traj, repeat, file_path=None):
    old_state = env.sim.get_state()
    old_qpos = old_state[1]
    old_qvel = old_state[2]

    # if file_path:
    #     from gym import wrappers
    #     env = wrappers.Monitor(env, file_path, force=True)
    if file_path:
        file_path = Path(file_path)
        frame_path = file_path / "frames"
        frame_path.mkdir(exist_ok=True, parents=True)

    env.reset()
    for i in range(len(traj)):
        qpos = traj[i].obs[:len(old_qpos)]
        qvel = traj[i].obs[len(old_qpos):len(old_qpos)+len(old_qvel)]
        # new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        # env.sim.set_state(new_state)
        # env.step(traj[i].ctrl)
        env.set_state(qpos, qvel)
        img_array = env.render(mode="rgb_array")
        img = Image.fromarray(img_array)
        img.save(frame_path / f"{i}.png")
        print(f"Rendered frame {i}")
        if not file_path:
            time.sleep(0.05)


    # Run ffmpeg to generate video
    frame_rate = env.metadata["video.frames_per_second"]
    mp4_path = file_path / "video.mp4"
    subprocess.call(["ffmpeg", "-r", f"{frame_rate}", "-f", "image2", "-s", "1920x1080", 
      "-stream_loop", f"{repeat}", "-i", f"{frame_path.resolve()}/%d.png", "-vcodec", "libx264", 
      "-pix_fmt", "yuv420p",
      f"{mp4_path.resolve()}"])

    env.close()

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
    _, total_reward, _, _, _  = env.step(u)
    # for _ in range(n_frames):
    #     _, reward, _, _, _ = env.sim.step()
    #     total_reward += reward

    return total_reward

def check_terminated(env, x, u):
    old_state = env.sim.get_state()
    old_qpos = old_state[1]
    old_qvel = old_state[2]

    qpos = x[:len(old_qpos)]
    qvel = x[len(old_qpos):len(old_qpos)+len(old_qvel)]

    # Represents a snapshot of the simulator's state.
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
    env.sim.set_state(new_state)

    _, _, terminated, _  = env.step(u)
    
    return terminated

class GymRewardCost(Cost):
    def __init__(self, system, env, cost_offset=200, plausible_threshold=-1000):
        Cost.__init__(self,system)
        self.env = env
        self._cost_offset = cost_offset
        self.plausible_threshold = plausible_threshold

    def __call__(self, traj, cost_offset=200):
        cum_reward = 0.0
        for i in range(len(traj)-1):
            cum_reward += gym_reward(self.env, traj[i].obs, traj[i].ctrl)
        cost = self._cost_offset - cum_reward
        if cost < self.plausible_threshold:
            cost = np.inf
        return cost

    def incremental(self,obs,ctrl):
        raise NotImplementedError

    def terminal(self,obs):
        raise NotImplementedError


def _get_init_obs(env):
    env.reset()
    qpos = env.sim.data.qpos
    qvel = env.sim.data.qvel
    return np.concatenate([qpos, qvel])


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
        env.reset(seed=0)
        self.env = env
        self.env_name = name
        system = self._get_system()

        task = Task(system)
        task.set_init_obs(self._get_init_obs())
        task.set_cost(self._get_cost(system))
        task.set_ctrl_bounds(env.action_space.low, env.action_space.high)
        task.set_num_steps(200)

        super().__init__(name, system, task, data_gen_method)

    def _get_system(self):
        state = self.env.sim.get_state()
        qpos = state[1]
        qvel = state[2]

        x_num = len(qpos) + len(qvel)
        u_num = self.env.action_space.shape[0]
        system = ampc.System([f"x{i}" for i in range(x_num)], [f"u{i}" for i in range(u_num)], self.env.dt)

        return system

    def _get_init_obs(self):
        return _get_init_obs(self.env)

    def _get_cost(self, system):
        return GymRewardCost(system, self.env)

    def dynamics(self, x, u):
        return gym_dynamics(self.env,x,u,n_frames=self.env.frame_skip)

    def gen_trajs(self, seed, n_trajs, traj_len=200, end_on_terminate=False):
        rng = np.random.default_rng(seed)
        trajs = []
        self.env.reset(seed=int(rng.integers(1 << 30)))
        self.env.action_space.seed(int(rng.integers(1 << 30)))
        
        for i in range(n_trajs):
            init_obs = self._get_init_obs()
            state = self.env.sim.get_state()
            traj = Trajectory.zeros(self.system, traj_len)
            traj[0].obs[:] = init_obs
                    
            for j in range(1, traj_len):
                action = self.env.action_space.sample()
                traj[j-1].ctrl[:] = action
                obs = self.dynamics(traj[j-1].obs[:], action)
                traj[j].obs[:] = obs

                if end_on_terminate:
                    if check_terminated(self.env, traj[j-1].obs, traj[j-1].ctrl):
                        traj = traj[:j]
                        break
            trajs.append(traj)

        return trajs

    def visualize(self, traj, repeat, file_path=None):
        """
        Visualize the half-cheetah trajectory using Gym functions.

        Parameters
        ----------
        traj : Trajectory
            Trajectory to visualize

        repeat : int
            Number of times to repeat trajectory in visualization

        file_path : str or Path
            Path to store video
        """
        viz_gym_traj(self.env, traj, repeat, file_path)

    @staticmethod
    def data_gen_methods():
        return ["uniform_random"]