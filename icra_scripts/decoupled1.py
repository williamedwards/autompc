# Created by William Edwards

# Standard library includes

# External project includes
import numpy as np
import torch
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from joblib import Memory, Parallel, delayed
import joblib
from pdb import set_trace
memory = Memory("cache")

# Internal project includes
import autompc as ampc
from autompc.evaluators import FixedSetEvaluator
from autompc.metrics import RmseKstepMetric
from autompc.sysid import MLP
from autompc.misc.model_metrics import *
from utils import save_result, load_result
import ConfigSpace as CS

default_cfg_vals = {
        "n_hidden_layers" : "2",
        "hidden_size_1" : 64,
        "hidden_size_2" : 64,
        "nonlintype" : "relu",
        "lr_log10" : -3
        }


@memory.cache
def train_mlp_inner(system, trajs, cfg_vals, torch_seed):
    cs = MLP.get_configuration_space(system)
    cfg = CS.Configuration(cs, cfg_vals)
    model = ampc.make_model(system, MLP, cfg, use_cuda=True)
    model.train(trajs)
    return model.get_parameters()

def train_mlp(system, trajs, cfg_vals=default_cfg_vals, torch_seed=0):
    cs = MLP.get_configuration_space(system)
    cfg = CS.Configuration(cs, cfg_vals)
    model = ampc.make_model(system, MLP, cfg, use_cuda=True)
    params = train_mlp_inner(system, trajs, cfg_vals, torch_seed)
    model.set_parameters(params)
    model.net = model.net.to("cpu")
    model._device = "cpu"
    return model

#@memory.cache
#def train_mlp_inner(system, trajs, torch_seed=0):
#    cs = MLP.get_configuration_space(system)
#    cfg = cs.get_default_configuration()
#    cfg["n_hidden_layers"] = "2"
#    cfg["hidden_size_1"] = 64
#    cfg["hidden_size_2"] = 64
#    #cfg["hidden_size_3"] = 128
#    model = ampc.make_model(system, MLP, cfg, use_cuda=False)
#    model.train(trajs)
#    return model.get_parameters()
#
#def train_mlp(system, trajs, torch_seed=0):
#    cs = MLP.get_configuration_space(system)
#    cfg = cs.get_default_configuration()
#    cfg["n_hidden_layers"] = "2"
#    cfg["hidden_size_1"] = 64
#    cfg["hidden_size_2"] = 64
#    #cfg["hidden_size_3"] = 128
#    model = ampc.make_model(system, MLP, cfg, use_cuda=False)
#    params = train_mlp_inner(system, trajs, torch_seed)
#    model.set_parameters(params)
#    model.net = model.net.to("cpu")
#    model._device = "cpu"
#    return model

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
        sim_traj = ampc.extend(sim_traj, [x], np.zeros((1, tinf.system.ctrl_dim)))
    return sim_traj

def run_smac(cs, eval_cfg, tune_iters, seed):
    rng = np.random.RandomState(seed)
    scenario = Scenario({"run_obj": "quality",  
                         "runcount-limit": tune_iters,  
                         "cs": cs,  
                         "deterministic": "true",
                         "limit_resources" : False,
                         "execdir" : "./smac"
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
    truedyn_costs = []
    for key, val in smac.runhistory.data.items():
        cfg = smac.runhistory.ids_config[key.config_id]
        if val.cost < inc_cost:
            inc_cost = val.cost
            inc_truedyn_cost = val.additional_info
            inc_cfg = cfg
        inc_costs.append(inc_cost)
        inc_truedyn_costs.append(inc_truedyn_cost)
        inc_cfgs.append(inc_cfg)
        cfgs.append(cfg)
        costs.append(val.cost)
        truedyn_costs.append(val.additional_info)
    ret_value["inc_costs"] = inc_costs
    ret_value["inc_truedyn_costs"] = inc_truedyn_costs
    ret_value["inc_cfgs"] = inc_cfgs
    ret_value["cfgs"] = cfgs
    ret_value["costs"] = costs
    ret_value["truedyn_costs"] = truedyn_costs

    return ret_value

def runexp_decoupled1(pipeline, tinf, tune_iters, ensemble_size, seed, 
        n_trajs, int_file=None, subexp=1):
    print(f"{seed=}")
    rng = np.random.default_rng(seed)
    sysid_trajs = tinf.gen_sysid_trajs(rng.integers(1 << 30), n_trajs=n_trajs)

    if subexp == 1 or subexp in [3,4]:
        root_pipeline_cfg = pipeline.get_configuration_space().get_default_configuration()
        root_pipeline_cfg["_model:n_hidden_layers"] = "3"
        root_pipeline_cfg["_model:hidden_size_1"] = 69
        root_pipeline_cfg["_model:hidden_size_2"] = 256
        root_pipeline_cfg["_model:hidden_size_3"] = 256
        root_pipeline_cfg["_model:lr_log10"] = -3.323534
        root_pipeline_cfg["_model:nonlintype"] = "tanh"
    elif subexp == 2:
        root_pipeline_cfg = pipeline.get_configuration_space().get_default_configuration()
        root_pipeline_cfg["_model:n_hidden_layers"] = "3"
        root_pipeline_cfg["_model:hidden_size_1"] = 128
        root_pipeline_cfg["_model:hidden_size_2"] = 128
        root_pipeline_cfg["_model:hidden_size_3"] = 128
    else:
        raise ValueError("Unrecognized sub experiment.")

    if subexp in [1,2]:
        surr_cfg_vals = default_cfg_vals
    elif subexp == 3:
        surr_cfg_vals = {
                "n_hidden_layers" : "3",
                "hidden_size_1" : 79,
                "hidden_size_2" : 235,
                "hidden_size_3" : 192,
                "lr_log10" : -3.1869328418567755,
                "nonlintype" : "sigmoid"
                }
    elif subexp == 4:
        surr_cfg_vals = {
                "n_hidden_layers" : "3",
                "hidden_size_1" : 139,
                "hidden_size_2" : 148,
                "hidden_size_3" : 162,
                "lr_log10" : -2.93579435372765,
                "nonlintype" : "sigmoid"
                }

    eval_seed = rng.integers(1 << 30)
    surrogates = []
    for _ in range(ensemble_size):
        surr_trajs = tinf.gen_surr_trajs(rng.integers(1 << 30), n_trajs=n_trajs)
        torch_seed = rng.integers(1 << 30)
        torch.manual_seed(torch_seed)
        surrogate = train_mlp(tinf.system, surr_trajs, surr_cfg_vals, torch_seed)
        surrogates.append(surrogate)
    print(f"{eval_seed=}")

    #cs = pipeline.get_configuration_space_fixed_model()
    #cfg = cs.get_default_configuration()
    #cfg["_controller:horizon"] = 25
    #cfg["_task_transformer_0:x_log10Qgain"] = 1.0
    #print(pipeline.set_configuration_fixed_model(root_pipeline_cfg, cfg))
    #set_trace()

    @memory.cache
    def train_model(sysid_trajs):
        model = ampc.make_model(tinf.system, pipeline.Model, 
                pipeline.get_model_cfg(root_pipeline_cfg), n_train_iters=50,
                use_cuda=False)
        torch.manual_seed(eval_seed)
        model.train(sysid_trajs)
        return model.get_parameters()
    model = ampc.make_model(tinf.system, pipeline.Model, 
            pipeline.get_model_cfg(root_pipeline_cfg), n_train_iters=5,
            use_cuda=True)
    model_params = train_model(sysid_trajs)
    model.set_parameters(model_params)
    def eval_cfg(cfg):
        torch.manual_seed(eval_seed)
        #pipeline_cfg = pipeline.set_tt_cfg(root_pipeline_cfg, 0, cfg)
        pipeline_cfg = pipeline.set_configuration_fixed_model(root_pipeline_cfg, cfg)
        controller, _ = pipeline(pipeline_cfg, sysid_trajs, model=model)
        surr_scores = []
        for surrogate in surrogates:
            surr_traj = runsim(tinf, 200, surrogate, controller)
            surr_score = tinf.perf_metric(surr_traj)
            surr_scores.append(surr_score)
        truedyn_traj = runsim(tinf, 200, None, controller, tinf.dynamics)
        truedyn_score = tinf.perf_metric(truedyn_traj)
        if not int_file is None:
            with open(int_file, "a") as f:
                print(cfg, file=f)
                print(f"Surrogate score is {surr_score}", file=f)
                print(f"True dynamics score is {truedyn_score}", file=f)
                print("==========\n\n", file=f)
        return max(surr_scores), (truedyn_score, surr_scores)

    #cs = pipeline.task_transformers[0].get_configuration_space(tinf.system)
    cs = pipeline.get_configuration_space_fixed_model()
    result = run_smac(cs, eval_cfg, tune_iters, rng.integers(1 << 30))
    baseline_res = eval_cfg(cs.get_default_configuration())
    return result, baseline_res

def runexp_decoupled2(pipeline, tinf, tune_iters, seed, int_file=None):
    print(f"{seed=}")
    rng = np.random.default_rng(seed)
    sysid_trajs = tinf.gen_sysid_trajs(rng.integers(1 << 30))
    surr_trajs = tinf.gen_surr_trajs(rng.integers(1 << 30))

    #sysid_torch_seeds = [rng.integers(1 << 30), rng.integers(1 << 30)]
    sysid_torch_seed = rng.integers(1 << 30)
    surr_torch_seeds = [rng.integers(1 << 30), rng.integers(1 << 30)]
    smac_seeds = [rng.integers(1 << 30), rng.integers(1 << 30)]

    surrogates = []
    for surr_torch_seed in surr_torch_seeds:
        torch.manual_seed(surr_torch_seed)
        surrogate = train_mlp(tinf.system, surr_trajs, surr_torch_seed)
        surrogates.append(surrogate)

    root_pipeline_cfg = pipeline.get_configuration_space().get_default_configuration()
    root_pipeline_cfg["_model:n_hidden_layers"] = "3"
    root_pipeline_cfg["_model:hidden_size_1"] = 69
    root_pipeline_cfg["_model:hidden_size_2"] = 256
    root_pipeline_cfg["_model:hidden_size_3"] = 256
    root_pipeline_cfg["_model:lr_log10"] = -3.323534
    root_pipeline_cfg["_model:nonlintype"] = "tanh"
    cs = pipeline.get_configuration_space_fixed_model()
    #cfg = cs.get_default_configuration()
    #cfg["_controller:horizon"] = 25
    #cfg["_task_transformer_0:x_log10Qgain"] = 1.0
    #print(pipeline.set_configuration_fixed_model(root_pipeline_cfg, cfg))
    #set_trace()

    @memory.cache
    def train_model(sysid_trajs):
        model = ampc.make_model(tinf.system, pipeline.Model, 
                pipeline.get_model_cfg(root_pipeline_cfg), n_train_iters=50,
                use_cuda=False)
        torch.manual_seed(sysid_torch_seed)
        model.train(sysid_trajs)
        return model.get_parameters()
    model = ampc.make_model(tinf.system, pipeline.Model, 
            pipeline.get_model_cfg(root_pipeline_cfg), n_train_iters=5,
            use_cuda=True)
    model_params = train_model(sysid_trajs)
    model.set_parameters(model_params)
    def eval_cfg(cfg, surr_idx):
        surrogate = surrogates[surr_idx]
        torch.manual_seed(sysid_torch_seed)
        #pipeline_cfg = pipeline.set_tt_cfg(root_pipeline_cfg, 0, cfg)
        pipeline_cfg = pipeline.set_configuration_fixed_model(root_pipeline_cfg, cfg)
        controller, _ = pipeline(pipeline_cfg, sysid_trajs, model=model)
        surr_traj = runsim(tinf, 200, surrogate, controller)
        surr_score = tinf.perf_metric(surr_traj)
        truedyn_traj = runsim(tinf, 200, None, controller, tinf.dynamics)
        truedyn_score = tinf.perf_metric(truedyn_traj)
        if not int_file is None:
            with open(int_file, "a") as f:
                print(cfg, file=f)
                print(f"Surrogate score is {surr_score}", file=f)
                print(f"True dynamics score is {truedyn_score}", file=f)
                print("==========\n\n", file=f)
        return surr_score, truedyn_score

    #cs = pipeline.task_transformers[0].get_configuration_space(tinf.system)
    cs = pipeline.get_configuration_space_fixed_model()
    results = []
    for surr_idx in range(len(surrogates)):
        for i, smac_seed in enumerate(smac_seeds):
            result = run_smac(cs, lambda cfg: eval_cfg(cfg, surr_idx), 
                    tune_iters, smac_seed)
            save_result(result, "decoupled2_int", seed, surr_idx, i)
            results.append(result)
    return results

#def get_model_rmse(model, trajs):
#    sqerrss = []
#    for traj in trajs:
#        preds = model.pred_parallel(traj.obs[:-1, :], traj.ctrls[:-1, :])
#        sqerrs = (preds - traj.obs[1:]) ** 2
#        sqerrss.append(sqerrs)
#    np.concatenate(sqerrss)
#    rmse = np.sqrt(np.mean(sqerrs, axis=-1))
#    return rmse
#
#def normalize(means, std, A):
#    At = []
#    for i in range(A.shape[1]):
#        At.append((A[:,i] - means[i]) / std[i])
#    return np.vstack(At).T
#
#def get_normalized_model_rmse(model, trajs, horiz=1):
#    dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
#    dy_means = np.mean(dY, axis=0)
#    dy_std = np.std(dY, axis=0)
#
#    sqerrss = []
#    for traj in trajs:
#        state = traj.obs[:-horiz, :]
#        for k in range(horiz):
#            pstate = state
#            state = model.pred_parallel(state, traj.ctrls[k:-(horiz-k), :])
#        pred_deltas = state - pstate
#        act_deltas = traj.obs[horiz:] - traj.obs[horiz-1:-1]
#        norm_pred_deltas = normalize(dy_means, dy_std, pred_deltas)
#        norm_act_deltas = normalize(dy_means, dy_std, act_deltas)
#        sqerrs = (norm_pred_deltas - norm_act_deltas) ** 2
#        sqerrss.append(sqerrs)
#    sqerrs = np.concatenate(sqerrss)
#    rmse = np.sqrt(np.mean(sqerrs, axis=None))
#    return rmse

@memory.cache(ignore=["tinf", "sysid_model"])
def get_tune_trajs(pipeline, tinf, sysid_model, load_args):
    def get_traj(cfg):
        root_pipeline_cfg = pipeline.get_configuration_space().get_default_configuration()
        pipeline_cfg = pipeline.set_configuration_fixed_model(root_pipeline_cfg, cfg)
        controller, _ = pipeline(pipeline_cfg, None, model=sysid_model)
        truedyn_traj = runsim(tinf, 200, None, controller, tinf.dynamics)
        return truedyn_traj
    result = load_result(*load_args)
    trajs = Parallel(n_jobs=8)(delayed(get_traj)(cfg) for cfg in result["cfgs"])
    return trajs

def dec2_surr_accuracy(pipeline, tinf, tune_iters, seed, int_file=None):
    rng = np.random.default_rng(seed)
    seed1 = rng.integers(1 << 30)
    seed2 = rng.integers(1 << 30)
    sysid_trajs = tinf.gen_sysid_trajs(seed1)
    sysid_trajs = tinf.gen_sysid_trajs(seed1)
    surr_trajs = tinf.gen_surr_trajs(seed2)
    surr_trajs = tinf.gen_surr_trajs(seed2)


    #sysid_torch_seeds = [rng.integers(1 << 30), rng.integers(1 << 30)]
    sysid_torch_seed = rng.integers(1 << 30)
    surr_torch_seeds = [rng.integers(1 << 30), rng.integers(1 << 30)]
    smac_seeds = [rng.integers(1 << 30), rng.integers(1 << 30)]

    seed3 = rng.integers(1 << 30)
    val_trajs = tinf.gen_sysid_trajs(seed3)
    val_trajs = tinf.gen_sysid_trajs(seed3)


    surrogates = []
    for surr_torch_seed in surr_torch_seeds:
        torch.manual_seed(surr_torch_seed)
        surrogate = train_mlp(tinf.system, surr_trajs, default_cfg_vals,
                surr_torch_seed)
        surrogates.append(surrogate)

    root_pipeline_cfg = pipeline.get_configuration_space().get_default_configuration()
    root_pipeline_cfg["_model:n_hidden_layers"] = "3"
    root_pipeline_cfg["_model:hidden_size_1"] = 69
    root_pipeline_cfg["_model:hidden_size_2"] = 256
    root_pipeline_cfg["_model:hidden_size_3"] = 256
    root_pipeline_cfg["_model:lr_log10"] = -3.323534
    root_pipeline_cfg["_model:nonlintype"] = "tanh"
    cs = pipeline.get_configuration_space_fixed_model()
    #cfg = cs.get_default_configuration()
    #cfg["_controller:horizon"] = 25
    #cfg["_task_transformer_0:x_log10Qgain"] = 1.0
    #print(pipeline.set_configuration_fixed_model(root_pipeline_cfg, cfg))
    #set_trace()

    @memory.cache
    def train_model(sysid_trajs):
        model = ampc.make_model(tinf.system, pipeline.Model, 
                pipeline.get_model_cfg(root_pipeline_cfg), n_train_iters=50,
                use_cuda=False)
        torch.manual_seed(sysid_torch_seed)
        model.train(sysid_trajs)
        return model.get_parameters()
    model = ampc.make_model(tinf.system, pipeline.Model, 
            pipeline.get_model_cfg(root_pipeline_cfg), n_train_iters=5,
            use_cuda=True)
    model_params = train_model(sysid_trajs)
    model.set_parameters(model_params)


    # Load in tune trajs
    load_args = ("decoupled2_int", 82, 0, 0)
    tune_trajs = get_tune_trajs(pipeline, tinf, model, load_args)

    for i, surrogate in enumerate(surrogates):
        surr_perf_err = get_model_metric_rmse(surrogate, surr_trajs, tinf.perf_metric)
        print("surr_perf_err=", surr_perf_err)
        surr_dev = get_model_variance([surrogate, model], surr_trajs)
        sysid_dev = get_model_variance([surrogate, model], sysid_trajs)
        val_dev = get_model_variance([surrogate, model], val_trajs)
        tune_dev = get_model_variance([surrogate, model], tune_trajs)
        print("surr_dev = ", surr_dev)
        print("sysid_dev = ", sysid_dev)
        print("val_dev = ", val_dev)
        print("tune_dev = ", tune_dev)
        for k in [1,5,10,15,20]:
            train_err = get_normalized_model_rmse(surrogate, surr_trajs, horiz=k)
            val_err = get_normalized_model_rmse(surrogate, sysid_trajs, horiz=k)
            tune_err = get_normalized_model_rmse(surrogate, tune_trajs, horiz=k)
            print("Surrogate {},{}-step: train_err={:.4f}, val_err={:.4f}, tune_err={:.4f}"
                    .format(i, k, train_err, val_err, tune_err))


def test_traj_hashing(pipeline, tinf, tune_iters, seed, int_file=None):
    surr_trajs = tinf.gen_sysid_trajs(306)
    surr_trajs2 = tinf.gen_sysid_trajs(306)
    print("hash1=", joblib.hash(tinf.system))
    print("hash2=", joblib.hash(surr_trajs))
    set_trace()
    surrogate = train_mlp(tinf.system, surr_trajs)