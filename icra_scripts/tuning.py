# Created by William Edwards (wre@illinois.edu), 2020-10-21

# Standard library includes
import os, sys
from pdb import set_trace
import argparse
import pickle

# External projects include
import numpy as np
import gym
#import gym_cartpole_swingup
import custom_gym_cartpole_swingup
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

# Internal project includes
import autompc as ampc

# Script includes
from tasks import init_task
from surrogates import init_surrogate
from pipelines import init_pipeline

def cartpole_simp_dynamics(y, u, g = 9.8, m = 1, L = 1, b = 0.1):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    theta, omega, x, dx = y
    return np.array([omega,
            g * np.sin(theta)/L - b * omega / (m*L**2) + u * np.cos(theta)/L,
            dx,
            u])

def dt_cartpole_dynamics(y,u,dt,g=9.8,m=1,L=1,b=1.0):
    y = np.copy(y)
    #y[0] += np.pi
    #sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    #if not sol.success:
    #    raise Exception("Integration failed due to {}".format(sol.message))
    #y = sol.y.reshape((4,))
    y += dt * cartpole_simp_dynamics(y,u[0],g,m,L,b)
    #y[0] -= np.pi
    return y



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
            tae_runner=lambda cfg: surrogate(pipeline, train_trajs, cfg))
    
    incumbent = smac.optimize()

    ret_value = dict()
    ret_value["incumbent"] = incumbent
    inc_cost = float("inf")
    inc_truedyn_cost = None
    inc_costs = []
    inc_truedyn_costs = []
    inc_cfgs = []
    inc_cfg = None
    costs_and_config_ids = []
    for key, val in smac.runhistory.data.items():
        if val.cost < inc_cost:
            inc_cost = val.cost
            inc_truedyn_cost = val.additional_info
            inc_cfg = smac.runhistory.ids_config[key.config_id]
        inc_costs.append(inc_cost)
        inc_truedyn_costs.append(inc_truedyn_cost)
        inc_cfgs.append(inc_cfg)
        costs_and_config_ids.append((val.cost, key.config_id))
    ret_value["inc_costs"] = inc_costs
    ret_value["inc_truedyn_costs"] = inc_costs
    ret_value["inc_cfgs"] = inc_cfgs
    costs_and_config_ids.sort()

    return ret_value



def main(args):
    task_info = init_task(args.task)
    if args.data == "buffer":
        trajs = load_buffer(task_info, args.bufferdir)
    else:
        raise
    sys.path.append("../examples/")
    from cartpole_control_test import trajs3
    for traj in trajs3:
        traj.ctrls /= 20.0
    trajs = trajs3
    #set_trace()
    train_trajs = trajs[:int(len(trajs)*args.datasplit)]
    surr_trajs = trajs[int(len(trajs)*args.datasplit):]
    surrogate = init_surrogate(task_info, args.simsteps, args.surrogate,
            surr_trajs)
    pipeline = init_pipeline(task_info, args.pipeline)
    #cs = pipeline.get_configuration_space()
    #cfg = cs.get_default_configuration()
    #cfg["_model:n_hidden_layers"] = "3"
    #cfg["_model:hidden_size_1"] = 128
    #cfg["_model:hidden_size_2"] = 128
    #cfg["_model:hidden_size_3"] = 128
    #surrogate(pipeline, train_trajs, cfg)
    ret_value = run_smac(pipeline, surrogate, train_trajs, args.seed, args.tuneiters)
    
    results_fn = f"{args.task}_{args.surrogate}_{args.seed}_{args.tuneiters}.pkl"
    results_path = os.path.join(args.resultsdir, results_fn)
    
    with open(results_path, "wb") as f:
        pickle.dump(ret_value, f)

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

