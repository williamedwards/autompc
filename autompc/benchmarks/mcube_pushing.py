# Created by William Edwards (wre2@illinois.edu), 2022-10-25

# Standard library includes
from pathlib import Path
from random import Random
import json

# Extenal library includes
import numpy as np

# Internal project includes
from ..system import System
from ..trajectory import Trajectory
from .modeling_benchmark import ModelingBenchmark

def _interp_poses(ts : np.ndarray, poses : np.ndarray):
    xs = np.interp(ts, poses[:, 0], poses[:, 1])
    ys = np.interp(ts, poses[:, 0], poses[:, 2])
    angles = np.interp(ts, poses[:, 0], poses[:, 3])

    return np.stack([xs, ys, angles], axis=1)

def _load_traj_from_path(path : Path, system : System):
    with open(path, "r") as f:
        traj_data = json.load(f)

    tip_data = np.array(traj_data["tip_pose"])
    obj_data = np.array(traj_data["object_pose"])

    ts = np.arange(tip_data[0,0], tip_data[-1,0], system.dt)
    tip_poses = _interp_poses(ts, tip_data)
    obj_poses = _interp_poses(ts, obj_data)

    observations = np.concatenate([tip_poses, obj_poses], axis=1)
    controls = tip_poses[1:, :] - tip_poses[:-1, :]
    controls = np.concatenate([controls, np.zeros((1,3))], axis=0) # Pad with zero control at end

    return Trajectory(system, observations, controls)

class MCubePushingBenchmark(ModelingBenchmark):
    def __init__(self, dt=0.005):
        system = System(["tip_x", "tip_y", "tip_angle", "obj_x", "obj_y", "obj_angle"],
            ["tip_dx", "tip_dy", "tip_dangle"], dt = dt)
        self.dt = dt
        self.input_paths = list((Path(__file__).parent / "data" / "mcube_pushing" / "rect1_json").glob("*.json"))

        super().__init__("MCubePushing", system)

    def get_trajs(self, num_trajs=None, shuffle_seed=0):
        """ Inherited, see superclass. """
        if num_trajs > self.max_num_trajs:
            raise ValueError("num_trajs cannot be greater than max_num_trajs")

        shuffle_rng = Random(shuffle_seed)
        selected_paths = shuffle_rng.sample(self.input_paths, num_trajs)

        trajs = [_load_traj_from_path(path, self.system) for path in selected_paths]

        return trajs

    @property
    def max_num_trajs(self):
        return len(self.input_paths)