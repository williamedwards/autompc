# Created by William Edwards (wre2@illinois.edu), 2020-12-10

# Standard library includes
import os, sys
import json
from pdb import set_trace
from collections import namedtuple

# Internal project includes
import autompc as ampc
from autompc.tasks import Task, QuadCost

# DALK includes
sys.path.append("/home/william/proj/microsurgery_data/src")
from experiment_helpers import *
from interpolator import LinearInterpolator
from extractors import *
from insertion import load_from_idx, load_from_yaml
from converter import InsertionTrajectoryConverter
from autoregression import AutoregModel

with open("/home/william/proj/microsurgery_data/datasets/Automatic2/missing.json") as f:
    bad_perception_data = json.load(f)

dalk = ampc.System(["x", "y", "pitch", "tx", "ty", "x-tx", "y-ty"], ["ux", "uy", "upitch"])
dalk.dt = 1.0

TaskInfo = namedtuple("TaskInfo", ["name", "system", "task", "init_obs", 
    "dynamics", "perf_metric", "gen_sysid_trajs", "gen_surr_trajs"])

def dalk_traj_to_ampc_traj(dalk_traj):
    ampc_traj = ampc.zeros(dalk, len(dalk_traj))
    for i, step in enumerate(dalk_traj):
        ampc_traj[i].obs[:2] = step.get_observations()
        ampc_traj[i].obs[2:-2] = step.get_features()
        ampc_traj[i].obs[-2] = ampc_traj[i, "x"] - ampc_traj[i, "tx"]
        ampc_traj[i].obs[-1] = ampc_traj[i, "y"] - ampc_traj[i, "ty"]
        ampc_traj[i].ctrl[:] = step.get_controls()
    return ampc_traj

def dalk_task():
    interp = LinearInterpolator()
    conv = InsertionTrajectoryConverter(extractors=[NeedlePitchExtractor(), 
        TargetExtractor()], 
        include_control_pitch=True, bad_perception_data=bad_perception_data,
        include_insertion_end=True)
    train = ['2', '4', '13', '16', '17', '24', '27', '30', '31', '33', '34',
            '38', '39', '41', '42', '46', '47', '54', '55']
    test = ['3', '9', '10', '11', '18', '21', '23', '26', '32', '35', '36',
            '40', '43', '44', '48', '53', '58', '64', '65']
    dalk_trajs1 = [get_traj_from_insertion(load_from_idx(idx), conv, interp) 
            for idx in train]
    dalk_trajs2 = [get_traj_from_insertion(load_from_idx(idx), conv, interp) 
            for idx in train]
    sysid_trajs = [dalk_traj_to_ampc_traj(traj) for traj in dalk_trajs1]
    surr_trajs = [dalk_traj_to_ampc_traj(traj) for traj in dalk_trajs2]
    gen_sysid_trajs = lambda s: sysid_trajs
    gen_surr_trajs = lambda s: surr_trajs
    Q = np.zeros((7,7))
    Q[-2,-2] = 1.0
    Q[-1,-1] = 1.0
    R = np.eye(3)
    F = np.zeros((7,7))
    F[-2,-2] = 1.0
    F[-1,-1] = 1.0
    cost = QuadCost(dalk, Q, R, F)
    task = Task(dalk)
    task.set_cost(cost)
    task.set_ctrl_bound("ux", -0.1, 0.1)
    task.set_ctrl_bound("uy", -0.1, 0.1)
    task.set_ctrl_bound("upitch", -0.1, 0.1)
    init_obs = sysid_trajs[0][0].obs
    def perf_metric(traj, threshold=0.2):
        cost = 0.0
        for i in range(len(traj)):
            if (np.abs(traj[i].obs[-1]) > threshold 
                    or np.abs(traj[i].obs[-2]) > threshold):
                cost += 1
        return cost

    return TaskInfo(name="DALK",
            system=dalk,
            task=task,
            init_obs=init_obs,
            dynamics=None,
            perf_metric=perf_metric,
            gen_sysid_trajs=gen_sysid_trajs,
            gen_surr_trajs=gen_surr_trajs)
