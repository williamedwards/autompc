# Created by William Edwards

from pdb import set_trace
import numpy as np
import autompc as ampc
from autompc.tasks.quad_cost import QuadCost
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from joblib import Memory
from func_timeout import func_timeout, FunctionTimedOut
memory = Memory("cache")
import time

import multiprocessing
multiprocessing.set_start_method("fork")
import torch
torch.set_num_threads(1)

rng = np.random.default_rng(43)

cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
dt = 0.01

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

    y += dt * cartpole_simp_dynamics(y,u[0],g,m,L,b)
    return y

cartpole.dt = dt

umin = -20.0
umax = 20.0
udmax = 0.25

# Generate trajectories for training
num_trajs = 500

@memory.cache
def gen_trajs(traj_len, num_trajs=num_trajs, dt=dt, seed=49):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(num_trajs):
        theta0 = rng.uniform(-0.002, 0.002, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(cartpole, traj_len)
        for i in range(traj_len):
            traj[i].obs[:] = y
            #if u[0] > umax:
            #    u[0] = umax
            #if u[0] < umin:
            #    u[0] = umin
            #u += rng.uniform(-udmax, udmax, 1)
            u  = rng.uniform(umin, umax, 1)
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs
trajs = gen_trajs(4)
trajs2 = gen_trajs(200)

Q = np.eye(4)
R = np.eye(1)

cost = QuadCost(cartpole, Q, R)

from autompc.tasks.task import Task

task = Task(cartpole)
task.set_cost(cost)
task.set_ctrl_bound("u", -20.0, 20.0)

from autompc.tasks.quad_cost_transformer import QuadCostTransformer
from autompc.pipelines import FixedControlPipeline
from autompc.sysid import Koopman, MLP
from autompc.control import FiniteHorizonLQR, IterativeLQR


#pipeline = FixedControlPipeline(cartpole, task, Koopman, FiniteHorizonLQR, 
#    [QuadCostTransformer])
pipeline = FixedControlPipeline(cartpole, task, MLP, IterativeLQR, 
    [QuadCostTransformer])

from autompc.control_evaluation import FixedModelEvaluator
from autompc.control_evaluation import FixedInitialMetric

init_states = [np.array([0.5, 0.0, 0.0, 0.0])]

metric = FixedInitialMetric(cartpole, task, init_states, sim_iters=100,
        sim_time_limit=10.0)

from autompc.sysid import ApproximateGaussianProcess, MLP
from cartpole_model import CartpoleModel
from noisy_cartpole_model import NoisyCartpoleModel


@memory.cache
def sample_approx_gp_inner(num_trajs, seed):
    cs = ApproximateGaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, ApproximateGaussianProcess, cfg,
            use_cuda=False)
    traj_sample = gen_trajs(traj_len=200, num_trajs=num_trajs, seed=seed)
    model.train(traj_sample)
    return model.get_parameters()

def sample_approx_gp(num_trajs, seed):
    cs = ApproximateGaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, ApproximateGaussianProcess, cfg,
            use_cuda=False)
    params = sample_approx_gp_inner(num_trajs, seed)
    model.set_parameters(params)
    model.gpmodel = model.gpmodel.to("cpu")
    model.device = "cpu"
    return model

@memory.cache
def sample_mlp_inner(num_trajs, seed):
    cs = MLP.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    cfg["n_hidden_layers"] = "3"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    cfg["hidden_size_3"] = 128
    model = ampc.make_model(cartpole, MLP, cfg, use_cuda=False)
    model.train(trajs2[-num_trajs:])
    return model.get_parameters()

def sample_mlp(num_trajs, seed):
    cs = MLP.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    cfg["n_hidden_layers"] = "3"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    cfg["hidden_size_3"] = 128
    model = ampc.make_model(cartpole, MLP, cfg, use_cuda=False)
    params = sample_mlp_inner(num_trajs, seed)
    model.set_parameters(params)
    model.net = model.net.to("cpu")
    model._device = "cpu"
    return model

# Generate surrogate models

true_dyn = CartpoleModel(cartpole)
low_noise = NoisyCartpoleModel(cartpole, np.random.default_rng(rng.integers(1 << 30)), 
        noise_factor=0.1)
high_noise = NoisyCartpoleModel(cartpole, np.random.default_rng(rng.integers(1 << 30)), 
        noise_factor=0.4)
low_data_gp = sample_approx_gp(10, 601)
high_data_gp = sample_approx_gp(80, 602)
import torch
low_data_mlp = sample_mlp(10, 603)
high_data_mlp = sample_mlp(80, 604)
#a = torch.zeros(100000)
#del a
#a = np.zeros(100000)
#print(a)

import torch.multiprocessing as tmul
import smac
smac.multiprocessing = tmul
#tmul.set_start_method("spawn")

surrogates = [("true_dyn", true_dyn),
        #("low_noise", low_noise),
        #("high_noise", high_noise),
        #("low_data_gp", low_data_gp),
        #("high_data_gp", high_data_gp),
        #("low_data_mlp", low_data_mlp),
        ("high_data_mlp", high_data_mlp)]

training_trajs = trajs[:]

evaluators = dict()
for label, model in surrogates:
    evaluator = FixedModelEvaluator(cartpole, task, metric, trajs2[:80], 
            sim_model=model)
    evaluators[label] = evaluator

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
@memory.cache
def run_smac(pipeline, label, seed, runcount_limit=5, n_jobs=1):
    evaluator = evaluators[label]
    rng = np.random.RandomState(seed)
    cs = pipeline.get_configuration_space()
    scenario = Scenario({"run_obj": "quality",  
                         "runcount-limit": runcount_limit,  
                         "cs": cs,  
                         "deterministic": "true",
                         })

    eval_cfg = evaluator(pipeline)
    #cfg = cs.get_default_configuration()
    #cfg["_controller:horizon"] = 500
    #cfg["_model:hidden_size"] = 88
    #cfg["_model:n_hidden_layers"] = 2
    #cfg["_model:n_train_iters"] = 30
    #cfg["_model:nonlintype"] = "relu"

    #cfg["_task_transformer_0:dx_log10Fgain"]
    #cfg["_task_transformer_0:dx_log10Qgain"]
    #cfg["_task_transformer_0:omega_log10Fgain"]
    #cfg["_task_transformer_0:omega_log10Qgain"]
    #cfg["_task_transformer_0:theta_log10Fgain"]
    #cfg["_task_transformer_0:theta_log10Qgain"]
    #cfg["_task_transformer_0:u_log10Rgain"]
    #cfg["_task_transformer_0:x_log10Fgain"]
    #cfg["_task_transformer_0:x_log10Qgain"]


    #eval_cfg(cs.get_default_configuration())
    #sys.exit(0)
    smac = SMAC4HPO(scenario=scenario, rng=rng,
            tae_runner=lambda cfg: eval_cfg(cfg),
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

evaluator_true = evaluators["true_dyn"]
@memory.cache
def reevealuate(smac_retval):
    new_scores = []
    eval_cfg = evaluator_true(pipeline)
    cfg_to_score = dict()
    for cfg in smac_retval["inc_cfgs"]:
        if cfg in cfg_to_score:
            score = cfg_to_score[cfg]
        else:
            score = eval_cfg(cfg)
            cfg_to_score[cfg] = score
        new_scores.append(score)
    return new_scores

rets = []
for label, _ in surrogates:
    smac_seed = rng.integers(1 << 30)
    ret = run_smac(pipeline, label, smac_seed, 
            runcount_limit=100)
    rets.append(ret)
from joblib import Parallel, delayed
#rets = Parallel(n_jobs=10)(delayed(run_smac)(pipeline, label,
#    rng.integers(1 << 30), runcount_limit=3) for label, _ in surrogates)

print("Evaluating true dynamics scores...")
#true_scores = [reevealuate(ret) for ret in rets]
true_scores = Parallel(n_jobs=10)(delayed(reevealuate)(ret) for ret in rets)
print(f"{true_scores=}")

#import sys
#sys.exit(0)
#
#set_trace()

fig = plt.figure()
ax = fig.gca()
for ret in rets:
    ax.plot(range(len(ret["inc_costs"])), ret["inc_costs"])
ax.set_title("Surrogate cost over time")
ax.set_xlabel("Tuning iterations")
ax.set_ylabel("Surrogate cost")
ax.legend([label for label, _ in surrogates])

fig = plt.figure()
ax = fig.gca()
for ts in true_scores:
    ax.plot(range(len(ts)), ts)
ax.set_title("True dynamics cost over time")
ax.set_xlabel("Tuning iterations")
ax.set_ylabel("True dynamics cost")
ax.legend([label for label, _ in surrogates])

plt.show()
