# Created by William Edwards

# Standard library includes

# External project includes
import numpy as np
import torch
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
#from smac.facade.roar_facade import ROAR
from smac.intensification.successive_halving import SuccessiveHalving
from joblib import Memory
from pdb import set_trace
memory = Memory("cache")

# Internal project includes
import autompc as ampc
import sys
from autompc.evaluators import FixedSetEvaluator
from autompc.metrics import RmseKstepMetric
from autompc.sysid import MLP, SINDy, LinearizedModel
from autompc.control import FiniteHorizonLQR, IterativeLQR, NonLinearMPC, MPPI, LQR
from autompc.tasks import QuadCost, Task
from autompc.tasks.quad_cost_transformer import QuadCostTransformer
from autompc.tasks.half_cheetah_transformer import HalfCheetahTransformer
from autompc.tasks.gaussain_reg_transformer import GaussianRegTransformer
from autompc.pipelines import FixedControlPipeline



@memory.cache
def train_mlp_inner(system, trajs, n_train_iters=50):
    cs = MLP.get_configuration_space(system)
    cfg = cs.get_default_configuration()
    cfg["n_hidden_layers"] = "3"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    cfg["hidden_size_3"] = 128
    model = ampc.make_model(system, MLP, cfg, use_cuda=True, n_train_iters=n_train_iters)
    model.train(trajs)
    return model.get_parameters()

def train_mlp(system, trajs, n_train_iters=50):
    cs = MLP.get_configuration_space(system)
    cfg = cs.get_default_configuration()
    cfg["n_hidden_layers"] = "3"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    cfg["hidden_size_3"] = 128
    model = ampc.make_model(system, MLP, cfg, use_cuda=True, n_train_iters=n_train_iters)
    params = train_mlp_inner(system, trajs, n_train_iters=n_train_iters)
    model.set_parameters(params)
    return model

def runsim(tinf, simsteps, sim_model, controller, dynamics=None):
    print("Enetering RunSim")
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

    intensifier_kwargs = {'initial_budget': 5, 'max_budget': 25, 'eta': 3,
                              'min_chall': 1, 'instance_order': 'shuffle_once'}

    smac = SMAC4HPO(scenario=scenario, rng=rng,
            tae_runner=eval_cfg) 
            #intensifier = SuccessiveHalving,
            #intensifier_kwargs = intensifier_kwargs,
            #n_jobs=10)
    
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

@memory.cache
def make_sysid(system, trajs):
    sysid_cfg = SINDy.get_configuration_space(system).get_default_configuration()
    sysid_cfg["trig_basis"] = "true"
    sysid_cfg["trig_interaction"] = "true"
    model = ampc.make_model(system, SINDy, sysid_cfg)
    model.train(trajs)
    return model


def runexp_controllers(pipeline, tinf, tune_iters, seed, simsteps, 
        controller_name, int_file=None, subexp=1):
    rng = np.random.default_rng(seed)
    sysid_trajs = tinf.gen_sysid_trajs(rng.integers(1 << 30))
    surr_trajs = tinf.gen_surr_trajs(rng.integers(1 << 30))

    torch.manual_seed(rng.integers(1 << 30))
    if subexp in [1,2]:
        surrogate = train_mlp(tinf.system, surr_trajs)
    elif subexp == 3:
        surrogate = train_mlp(tinf.system, surr_trajs, n_train_iters=1)
    eval_seed = rng.integers(1 << 30)

    if tinf.name=="HalfCheetah":
        if subexp != 3:
            model = train_mlp(tinf.system, sysid_trajs)
        else:
            model = train_mlp(tinf.system, sysid_trajs, n_train_iters=1)
        print("Using MLP SysID")
    else:
        print("Using SINDy SysID")
        model = make_sysid(tinf.system, sysid_trajs)
    if controller_name == "ilqr":
        Controller = IterativeLQR
    elif controller_name == "lqr":
        Controller = LQR
        model = LinearizedModel(tinf.system, np.zeros(tinf.system.obs_dim),
                model)
    elif controller_name == "dt":
        Controller = NonLinearMPC
    elif controller_name == "mppi":
        Controller = MPPI


    # # Initialize tuned objecive function
    # if tinf.name=="Pendulum-Swingup":
    #     Q = np.eye(2)
    #     R = 0.001 * np.eye(1)
    #     F = np.eye(2)
    # elif tinf.name=="CartPole-Swingup":
    #     Q = np.eye(4)
    #     R = 0.01 * np.eye(1)
    #     F = 10.0 * np.eye(4)
    # cost = QuadCost(tinf.system, Q, R, F)
    # task = tinf.task
    # task.set_cost(cost)

    # Initialize pipeline
    if tinf.name=="CartPole-Swingup" and subexp==1:
        pipeline = FixedControlPipeline(tinf.system, tinf.task, MLP, 
                Controller, [QuadCostTransformer])
    elif tinf.name=="HalfCheetah" and subexp in [1,3]:
        pipeline = FixedControlPipeline(tinf.system, tinf.task, MLP, 
                Controller, [HalfCheetahTransformer])
    elif tinf.name=="HalfCheetah" and subexp in 2:
        pipeline = FixedControlPipeline(tinf.system, tinf.task, MLP, 
                Controller, [HalfCheetahTransformer, GaussianRegTransformer])
    elif tinf.name=="Pendulum-Swingup" and subexp==1:
        pipeline = FixedControlPipeline(tinf.system, tinf.task, MLP, 
                Controller, [QuadCostTransformer])
    elif tinf.name=="Pendulum-Swingup" and subexp==2:
        pipeline = FixedControlPipeline(tinf.system, tinf.task, MLP, 
                Controller, [QuadCostTransformer, GaussianRegTransformer])
    elif tinf.name=="CartPole-Swingup" and subexp==2:
        pipeline = FixedControlPipeline(tinf.system, tinf.task, MLP, 
                Controller, [QuadCostTransformer, GaussianRegTransformer])
    else:
        set_trace()
        raise ValueError("Unhandled case")

    root_pipeline_cfg = pipeline.get_configuration_space().get_default_configuration()


    def eval_cfg(cfg):
        print("Making controller")
        pipeline_cfg = pipeline.set_configuration_fixed_model(root_pipeline_cfg, cfg)
        controller, _ = pipeline(pipeline_cfg, sysid_trajs, model=model)
        print("Entering simulation")
        surr_traj = runsim(tinf, simsteps, surrogate, controller)
        truedyn_traj = runsim(tinf, simsteps, None, controller, tinf.dynamics)
        surr_score = tinf.perf_metric(surr_traj)
        truedyn_score = tinf.perf_metric(truedyn_traj)
        if not int_file is None:
            with open(int_file, "a") as f:
                print(cfg, file=f)
                print(f"Surrogate score is {surr_score}", file=f)
                print(f"True dynamics score is {truedyn_score}", file=f)
                print("==========\n\n", file=f)
        return surr_score, truedyn_score

    # Debug
    #cs = Controller.get_configuration_space(tinf.system, task, model)
    cs = pipeline.get_configuration_space_fixed_model()
    #cfg = cs.get_default_configuration()
    ##cfg["finite_horizon"] = "false"
    #eval_cfg(cfg)
    #sys.exit(0)

    result = run_smac(cs, eval_cfg, tune_iters, rng.integers(1 << 30))
    return result
