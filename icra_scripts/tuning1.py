# Created by William Edwards

# Standard library includes
from pdb import set_trace
import time

# External project includes
import numpy as np
import torch
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from joblib import Memory
memory = Memory("cache")

# Internal project includes
import autompc as ampc
from autompc.evaluators import FixedSetEvaluator
from autompc.metrics import RmseKstepMetric
from autompc.sysid import MLP

from swimmer_task import viz_swimmer_traj
from utils import load_result

@memory.cache
def train_mlp_inner(system, trajs, n_train_iters):
    cs = MLP.get_configuration_space(system)
    cfg = cs.get_default_configuration()
    cfg["n_hidden_layers"] = "3"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    cfg["hidden_size_3"] = 128
    model = ampc.make_model(system, MLP, cfg, use_cuda=True,
            n_train_iters=n_train_iters)
    model.train(trajs)
    return model.get_parameters()

def train_mlp(system, trajs, n_train_iters=50):
    cs = MLP.get_configuration_space(system)
    cfg = cs.get_default_configuration()
    cfg["n_hidden_layers"] = "3"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    cfg["hidden_size_3"] = 128
    model = ampc.make_model(system, MLP, cfg, use_cuda=True,
            n_train_iters=n_train_iters)
    params = train_mlp_inner(system, trajs, n_train_iters)
    model.set_parameters(params)
    return model

def runsim(tinf, simsteps, sim_model, controller, dynamics=None):
    sim_traj = ampc.zeros(tinf.system, 1)
    x = np.copy(tinf.init_obs)
    sim_traj[0].obs[:] = x
    
    constate = controller.traj_to_state(sim_traj)
    if dynamics is None:
        simstate = sim_model.traj_to_state(sim_traj)
    for _  in range(simsteps):
        u, constate = controller.run(constate, sim_traj[-1].obs)
        if dynamics is None:
            simstate = sim_model.pred(simstate, u)
            x = simstate[:tinf.system.obs_dim]
        else:
            x = dynamics(x, u)
        print(f"{u=} {x=}")
        sim_traj[-1].ctrl[:] = u
        sim_traj = ampc.extend(sim_traj, [x], 
                np.zeros((1, tinf.system.ctrl_dim)))
    return sim_traj

def run_smac(cs, eval_cfg, tune_iters, seed):
    rng = np.random.RandomState(seed)
    scenario = Scenario({"run_obj": "quality",  
                         "runcount-limit": tune_iters,  
                         "cs": cs,  
                         "deterministic": "true",
                         "execdir" : "./smac",
                         "limit_resources" : False
                         })

    smac = SMAC4HPO(scenario=scenario, rng=rng,
            tae_runner=eval_cfg)
    
    incumbent = smac.optimize()

    ret_value = dict()
    ret_value["incumbent"] = incumbent
    inc_cost = float("inf")
    inc_truedyn_cost = None
    inc_costs = []
    inc_truedyn_costs = []
    inc_cfgs = []
    inc_cfg = None
    cfgs = []
    costs = []
    surr_scores = []
    truedyn_costs = []
    addinfos = []
    surr_trajs = []
    truedyn_trajs = []
    for key, val in smac.runhistory.data.items():
        cfg = smac.runhistory.ids_config[key.config_id]
        if val.cost < inc_cost:
            inc_cost = val.cost
            inc_truedyn_cost = val.additional_info["truedyn_score"]
            inc_cfg = cfg
        inc_costs.append(inc_cost)
        inc_truedyn_costs.append(inc_truedyn_cost)
        inc_cfgs.append(inc_cfg)
        cfgs.append(cfg)
        costs.append(val.cost)
        truedyn_costs.append(val.additional_info["truedyn_score"])
        surr_trajs.append(val.additional_info["surr_traj"])
        truedyn_trajs.append(val.additional_info["truedyn_traj"])
        surr_scores.append(val.additional_info["surr_score"])
        addinfos.append(val.additional_info)
    ret_value["inc_costs"] = inc_costs
    ret_value["inc_truedyn_costs"] = inc_truedyn_costs
    ret_value["inc_cfgs"] = inc_cfgs
    ret_value["cfgs"] = cfgs
    ret_value["costs"] = costs
    ret_value["truedyn_costs"] = truedyn_costs
    ret_value["addinfos"] = addinfos
    ret_value["surr_scores"] = surr_scores
    ret_value["surr_trajs"] = surr_trajs
    ret_value["truedyn_trajs"] = truedyn_trajs

    return ret_value

def runexp_tuning1(pipeline, tinf, tune_iters, seed, simsteps, int_file=None, subexp=1):
    rng = np.random.default_rng(seed)
    sysid_trajs = tinf.gen_sysid_trajs(rng.integers(1 << 30))
    surr_trajs = tinf.gen_surr_trajs(rng.integers(1 << 30))

    torch.manual_seed(rng.integers(1 << 30))
    if subexp == 1:
        surrogate = train_mlp(tinf.system, surr_trajs)
    elif subexp == 2 or 3:
        surrogate = train_mlp(tinf.system, surr_trajs, n_train_iters=500)
    eval_seed = rng.integers(1 << 30)
    print(f"{eval_seed=}")

    def eval_cfg(cfg):
        start_time = time.time()
        torch.manual_seed(eval_seed)
        controller, model = pipeline(cfg, sysid_trajs)
        sysid_time = time.time() - start_time
        start_time = time.time()
        surr_traj = runsim(tinf, simsteps, surrogate, controller)
        surr_traj_time = time.time() - start_time
        start_time = time.time()
        if tinf.dynamics is not None:
            truedyn_traj = runsim(tinf, simsteps, None, controller, tinf.dynamics)
            truedyn_traj_time = time.time() - start_time
            truedyn_score = tinf.perf_metric(truedyn_traj)
        else:
            truedyn_score = np.nan
            truedyn_traj_time = 0.0
            truedyn_traj = None
        surr_score = tinf.perf_metric(surr_traj)
        if subexp != 3 or surr_score > -100.0:
            score = surr_score
        else:
            score = 200
        if not int_file is None:
            with open(int_file, "a") as f:
                print(cfg, file=f)
                print(f"Surrogate score is {surr_score}", file=f)
                print(f"True dynamics score is {truedyn_score}", file=f)
                print(f"Score is {score}", file=f)
                print("==========\n\n", file=f)
        return score, {"truedyn_score" : truedyn_score,
                "sysid_time" : sysid_time,
                "surr_traj" : (surr_traj.obs.tolist(), surr_traj.ctrls.tolist()),
                "truedyn_traj" : (truedyn_traj.obs.tolist(), truedyn_traj.ctrls.tolist()),
                "surr_traj_time" : surr_traj_time,
                "truedyn_traj_time" : truedyn_traj_time,
                "surr_score" : surr_score}

    #cfg1 = pipeline.get_configuration_space().get_default_configuration()
    #cfg1["_task_transformer_0:u_log10Rgain"] = -2
    #cfg1["_task_transformer_0:theta_log10Fgain"] = 2 
    #cfg1["_task_transformer_0:omega_log10Fgain"] = 2 
    #cfg1["_task_transformer_0:x_log10Fgain"] = 2 
    #cfg1["_task_transformer_0:dx_log10Fgain"] = 2 
    #cfg1["_controller:horizon"] = 20
    #cfg1["_model:n_hidden_layers"] = "3"
    #cfg1["_model:hidden_size_1"] = 128
    #cfg1["_model:hidden_size_2"] = 128
    #cfg1["_model:hidden_size_3"] = 128

    result = run_smac(pipeline.get_configuration_space(), eval_cfg, 
            tune_iters, rng.integers(1 << 30))
    return result

def tuning1_viz(pipeline, tinf, tune_iters, seed, simsteps, int_file=None, subexp=1, args=None):
    rng = np.random.default_rng(seed)
    sysid_trajs = tinf.gen_sysid_trajs(rng.integers(1 << 30))
    surr_trajs = tinf.gen_surr_trajs(rng.integers(1 << 30))


    torch.manual_seed(rng.integers(1 << 30))
    #if subexp == 1:
        #surrogate = train_mlp(tinf.system, surr_trajs)
    #elif subexp == 2:
        #surrogate = train_mlp(tinf.system, surr_trajs, n_train_iters=500)
    eval_seed = rng.integers(1 << 30)
    print(f"{eval_seed=}")

    def viz_cfg(cfg):
        start_time = time.time()
        torch.manual_seed(eval_seed)
        controller, model = pipeline(cfg, sysid_trajs)
        sysid_time = time.time() - start_time
        start_time = time.time()
        #surr_traj = runsim(tinf, simsteps, surrogate, controller)
        #surr_traj_time = time.time() - start_time
        start_time = time.time()
        if tinf.dynamics is not None:
            truedyn_traj = runsim(tinf, simsteps, None, controller, tinf.dynamics)
            truedyn_traj_time = time.time() - start_time
            truedyn_score = tinf.perf_metric(truedyn_traj)
        else:
            truedyn_score = np.nan
            truedyn_traj_time = 0.0
        #surr_score = tinf.perf_metric(surr_traj)
        set_trace()
        viz_swimmer_traj(truedyn_traj, repeat=1000)

    result = load_result("tuning1", args.task, args.pipeline, args.subexp,
                args.tuneiters, args.seed) 
    cfg = result["inc_cfgs"][-1]
    viz_cfg(cfg)
