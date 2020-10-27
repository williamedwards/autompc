# Created by William Edwards

# Standard library includes
from pdb import set_trace

# External project includes
import numpy as np
import torch
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

# Internal project includes
import autompc as ampc
from autompc.evaluators import FixedSetEvaluator
from autompc.graphs import KstepGrapher
from autompc.metrics import RmseKstepMetric
from tuning1 import runsim, train_mlp

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
    inc_model_params = None
    inc_costs = []
    inc_truedyn_costs = []
    inc_model_seeds = []
    inc_cfgs = []
    inc_cfg = None
    costs_and_config_ids = []
    for key, val in smac.runhistory.data.items():
        if val.cost < inc_cost:
            inc_cost = val.cost
            inc_truedyn_cost = val.additional_info[0]
            inc_model_seed = val.additional_info[1]
            inc_cfg = smac.runhistory.ids_config[key.config_id]
        inc_costs.append(inc_cost)
        inc_truedyn_costs.append(inc_truedyn_cost)
        inc_model_seeds.append(inc_model_seed)
        inc_cfgs.append(inc_cfg)
        costs_and_config_ids.append((val.cost, key.config_id))
    ret_value["inc_costs"] = inc_costs
    ret_value["inc_truedyn_costs"] = inc_truedyn_costs
    ret_value["inc_model_seeds"] = inc_model_seeds
    ret_value["inc_cfgs"] = inc_cfgs
    costs_and_config_ids.sort()

    return ret_value


def runexp_sysid2(pipeline, tinf, tune_iters, sub_exp, seed):
    rng = np.random.default_rng(seed)
    sysid_trajs = tinf.gen_sysid_trajs(rng.integers(1 << 30))
    surr_trajs = tinf.gen_surr_trajs(rng.integers(1 << 30))
    training_set = sysid_trajs[:int(0.7*len(sysid_trajs))]
    validation_set = sysid_trajs[int(0.7*len(sysid_trajs)):int(0.85*len(sysid_trajs))]
    testing_set = sysid_trajs[int(0.85*len(sysid_trajs)):]
    surrogate = train_mlp(tinf.system, surr_trajs)
    
    metric1 = RmseKstepMetric(tinf.system, k=1)
    metric2 = RmseKstepMetric(tinf.system, k=int(1/tinf.system.dt))
    tuning_evaluator1 = FixedSetEvaluator(tinf.system, training_set + validation_set, 
            metric1, rng, training_trajs=training_set)
    tuning_evaluator2 = FixedSetEvaluator(tinf.system, training_set + validation_set, 
            metric2, rng, training_trajs=training_set)
    final_evaluator = FixedSetEvaluator(tinf.system, training_set + testing_set, 
            metric2, rng, training_trajs=training_set)
    kstep_grapher = KstepGrapher(tinf.system, kmax=int(3/tinf.system.dt), kstep=2,
            evalstep=5)
    final_evaluator.add_grapher(kstep_grapher)

    root_pipeline_cfg = pipeline.get_configuration_space().get_default_configuration()
    eval_seed1 = int(rng.integers(1 << 30))
    def eval_cfg1(cfg):
        torch.manual_seed(eval_seed1)
        score, _, _, model = tuning_evaluator1(pipeline.Model, cfg,
                ret_trained_model=True)
        pipeline_cfg = pipeline.set_model_cfg(root_pipeline_cfg, cfg)
        controller, _ = pipeline(pipeline_cfg, sysid_trajs, model)
        print("startrunsim")
        truedyn_traj = runsim(tinf, 200, None, controller, tinf.dynamics)
        print("donerunsim")
        truedyn_score = tinf.perf_metric(truedyn_traj)
        print("donemetric")
        model_params = model.get_parameters()
        additional_info = (truedyn_score, model.get_parameters())
        print("got additional_info")
        return score, (truedyn_score, eval_seed1)

    eval_seed2 = int(rng.integers(1 << 30))
    def eval_cfg2(cfg):
        torch.manual_seed(eval_seed2)
        score, _, _, model = tuning_evaluator2(pipeline.Model, cfg,
                ret_trained_model=True)
        pipeline_cfg = pipeline.set_model_cfg(root_pipeline_cfg, cfg)
        controller, _ = pipeline(pipeline_cfg, sysid_trajs, model)
        truedyn_traj = runsim(tinf, 200, None, controller, tinf.dynamics)
        truedyn_score = tinf.perf_metric(truedyn_traj)
        return score, (truedyn_score, eval_seed2)

    eval_seed3 = int(rng.integers(1 << 30))
    def eval_cfg3(cfg):
        torch.manual_seed(eval_seed3)
        controller, model = pipeline(cfg, sysid_trajs)
        surr_traj = runsim(tinf, 200, surrogate, controller)
        truedyn_traj = runsim(tinf, 200, None, controller, tinf.dynamics)
        surr_score = tinf.perf_metric(surr_traj)
        truedyn_score = tinf.perf_metric(truedyn_traj)
        return surr_score, (truedyn_score, eval_seed3)

    if sub_exp == 1:
        result = run_smac(pipeline.Model.get_configuration_space(tinf.system),
                eval_cfg1, tune_iters, rng.integers(1 << 30))
    elif sub_exp == 2:
        result = run_smac(pipeline.Model.get_configuration_space(tinf.system),
                eval_cfg2, tune_iters, rng.integers(1 << 30))
    elif sub_exp == 3:
        result = run_smac(pipeline.get_configuration_space(),
                eval_cfg3, tune_iters, rng.integers(1 << 30))

    #inc_model_params = result["inc_model_paramss"][-1]
    inc_cfg = result["inc_cfgs"][-1]
    if sub_exp == 3:
        inc_cfg = pipeline.get_model_cfg(inc_cfg)
    #inc_model = ampc.make_model(tinf.system, pipeline.Model, inc_cfg)
    #inc_model.set_parameters(inc_model_params)
    inc_seed = result["inc_model_seeds"][-1]

    torch.manual_seed(inc_seed)
    _, _, graphs = final_evaluator(pipeline.Model, inc_cfg, use_cuda=False)
    rmses, _, horizs = graphs[0].get_rmses()

    return result, (rmses, horizs)
