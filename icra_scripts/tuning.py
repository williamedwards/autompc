# Created by William Edwards (wre@illinois.edu), 2020-10-21

# Standard library includes
import os, sys
from pdb import set_trace
import argparse

# External projects include
import numpy as np
import gym
import gym_cartpole_swingup

# Internal project includes
import autompc as ampc

# Script includes
from tasks import init_task
from surrogates import init_surrogate
from pipelines import init_pipeline

def load_buffer(tinf, buffer_dir, prefix=None):
    if prefix is None:
        prefix = "Robust_" + tinf.env_name + "_0_"
    states = np.load(os.path.join(buffer_dir, prefix+"state.npy"))
    actions = np.load(os.path.join(buffer_dir, prefix+"action.npy"))
    next_states = np.load(os.path.join(buffer_dir, prefix+"next_state.npy"))

    episode_start = 0
    trajs = []
    for i in range(states.shape[0]):
        if i == states.shape[0]-1 or (next_states[i] != states[i+1]).any():
            traj = ampc.empty(tinf.system, i - episode_start + 1)
            traj.obs[:] = np.apply_along_axis(tinf.gym_to_obs, 1, 
                    states[episode_start:i+1])
            traj.ctrls[:] = np.apply_along_axis(tinf.gym_to_ctrl, 1, 
                    actions[episode_start:i+1])
            trajs.append(traj)
            episode_start = i+1
    return trajs

def run_smac(pipeline, surrogate, train_trajs, seed, tuneiters):
    rng = np.random.RandomState(seed)
    cs = pipeline.get_configuration_space()
    scenario = Scenario({"run_obj": "quality",  
                         "runcount-limit": tuneiters,  
                         "cs": cs,  
                         "deterministic": "true",
                         })

    smac = SMAC4HPO(scenario=scenario, rng=rng,
            tae_runner=lambda cfg: -surrogate(pipeline, train_trajs, cfg)),
            n_jobs=1)
    
    incumbent = smac.optimize()

    ret_value = dict()
    ret_value["incumbent"] = incumbent
    inc_cost = float("inf")
    inc_costs = []
    inc_cfgs = []
    inc_cfg = None
    costs_and_config_ids = []
    for key, val in smac.runhistory.data.items():
        if val.cost < inc_cost:
            inc_cost = val.cost
            inc_cfg = smac.runhistory.ids_config[key.config_id]
        inc_costs.append(inc_cost)
        inc_cfgs.append(inc_cfg)
        costs_and_config_ids.append((val.cost, key.config_id))
    ret_value["inc_costs"] = inc_costs
    ret_value["inc_cfgs"] = inc_cfgs
    costs_and_config_ids.sort()
    top_five = [(smac.runhistory.ids_config[cfg_id], cost) for cost, cfg_id 
        in costs_and_config_ids[:5]]
    ret_value["top_five"] = top_five

    return ret_value



def main(args):
    task_info = init_task(args.task)
    if args.data == "buffer":
        trajs = load_buffer(task_info, args.bufferdir)
    else:
        raise
    train_trajs = trajs[:int(len(trajs)*args.datasplit)]
    surr_trajs = trajs[int(len(trajs)*args.datasplit):]
    surrogate = init_surrogate(task_info, args.simsteps, args.surrogate,
            surr_trajs)
    pipeline = init_pipeline(args.pipeline)
    ret_value = run_smac(pipeline, surrogate, train_trajs, args.seed, args.tuneiters)
    set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--task", default="cartpole-swingup")
    parser.add_argument("--pipeline", default="mlp-ilqr")
    parser.add_argument("--data", default="buffer")
    parser.add_argument("--datasplit", default=0.5, type=float)
    parser.add_argument("--bufferdir", default="buffers")
    parser.add_argument("--resultsdir", default="results")
    parser.add_argument("--surrogate", default="truedyn",
            choices=["truedyn", "gp", "mlp", "boot-mlp"])
    parser.add_argument("--tuneiters", type=int, default=10)
    parser.add_argument("--simsteps", type=int, default=1000)
    args = parser.parse_args()
    main(args)

