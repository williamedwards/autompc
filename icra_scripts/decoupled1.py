# Created by William Edwards

# Standard library includes

# External project includes
import numpy as np
import torch
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from joblib import Memory
from pdb import set_trace
memory = Memory("cache")

# Internal project includes
import autompc as ampc
from autompc.evaluators import FixedSetEvaluator
from autompc.metrics import RmseKstepMetric
from autompc.sysid import MLP


@memory.cache
def train_mlp_inner(system, trajs):
    cs = MLP.get_configuration_space(system)
    cfg = cs.get_default_configuration()
    cfg["n_hidden_layers"] = "3"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    cfg["hidden_size_3"] = 128
    model = ampc.make_model(system, MLP, cfg, use_cuda=False)
    model.train(trajs)
    return model.get_parameters()

def train_mlp(system, trajs):
    cs = MLP.get_configuration_space(system)
    cfg = cs.get_default_configuration()
    cfg["n_hidden_layers"] = "3"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    cfg["hidden_size_3"] = 128
    model = ampc.make_model(system, MLP, cfg, use_cuda=False)
    params = train_mlp_inner(system, trajs)
    model.set_parameters(params)
    model.net = model.net.to("cpu")
    model._device = "cpu"
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
        sim_traj = ampc.extend(sim_traj, [x], np.zeros((1, tinf.system.ctrl_dim)))
    return sim_traj

def run_smac(cs, eval_cfg, tune_iters, seed):
    rng = np.random.RandomState(seed)
    scenario = Scenario({"run_obj": "quality",  
                         "runcount-limit": tune_iters,  
                         "cs": cs,  
                         "deterministic": "true",
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

def runexp_decoupled1(pipeline, tinf, tune_iters, seed, int_file=None,
        subexp=1):
    rng = np.random.default_rng(seed)
    sysid_trajs = tinf.gen_sysid_trajs(rng.integers(1 << 30))
    surr_trajs = tinf.gen_surr_trajs(rng.integers(1 << 30))

    torch.manual_seed(rng.integers(1 << 30))
    surrogate = train_mlp(tinf.system, surr_trajs)
    eval_seed = rng.integers(1 << 30)

    if subexp == 1:
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
            use_cuda=False)
    model_params = train_model(sysid_trajs)
    model.set_parameters(model_params)
    def eval_cfg(cfg):
        torch.manual_seed(eval_seed)
        #pipeline_cfg = pipeline.set_tt_cfg(root_pipeline_cfg, 0, cfg)
        pipeline_cfg = pipeline.set_configuration_fixed_model(root_pipeline_cfg, cfg)
        controller, _ = pipeline(pipeline_cfg, sysid_trajs, model=model)
        surr_traj = runsim(tinf, 200, surrogate, controller)
        truedyn_traj = runsim(tinf, 200, None, controller, tinf.dynamics)
        surr_score = tinf.perf_metric(surr_traj)
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
    result = run_smac(cs, eval_cfg, tune_iters, rng.integers(1 << 30))
    baseline_res = eval_cfg(cs.get_default_configuration())
    return result, baseline_res
