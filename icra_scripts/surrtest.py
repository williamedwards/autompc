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
    model = ampc.make_model(system, MLP, cfg, use_cuda=True)
    model.train(trajs)
    return model.get_parameters()

def train_mlp(system, trajs):
    cs = MLP.get_configuration_space(system)
    cfg = cs.get_default_configuration()
    cfg["n_hidden_layers"] = "3"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    cfg["hidden_size_3"] = 128
    model = ampc.make_model(system, MLP, cfg, use_cuda=True)
    params = train_mlp_inner(system, trajs)
    model.set_parameters(params)
    #model.net = model.net.to("cpu")
    #model._device = "cpu"
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

def runexp_surrtest(pipeline, tinf, tune_iters, seed, simsteps, int_file=None):
    rng = np.random.default_rng(seed)
    sysid_trajs = tinf.gen_sysid_trajs(rng.integers(1 << 30))

    rng.integers(1 << 30)
    torch.manual_seed(rng.integers(1 << 30))
    eval_seed = rng.integers(1 << 30)

    surrogates = []
    for i in range(10):
        surr_trajs = tinf.gen_surr_trajs(rng.integers(1 << 30))
        surrogate = train_mlp(tinf.system, surr_trajs)
        surrogates.append(surrogate)

    def eval_cfg(controller, surrogate, eval_true=True):
        surr_traj = runsim(tinf, simsteps, surrogate, controller)
        if eval_true:
            truedyn_traj = runsim(tinf, simsteps, None, controller, tinf.dynamics)
            truedyn_score = tinf.perf_metric(truedyn_traj)
        surr_score = tinf.perf_metric(surr_traj)
        if not int_file is None:
            with open(int_file, "a") as f:
                print(cfg, file=f)
                print(f"Surrogate score is {surr_score}", file=f)
                print(f"True dynamics score is {truedyn_score}", file=f)
                print("==========\n\n", file=f)
        if eval_true:
            return surr_score, truedyn_score
        else:
            return surr_score

    cfg1 = pipeline.get_configuration_space().get_default_configuration()
    cfg1["_controller:horizon"] = 19
    cfg1["_model:n_hidden_layers"] = '4'
    cfg1["_model:hidden_size_1"] = 196
    cfg1["_model:hidden_size_2"] = 196
    cfg1["_model:hidden_size_3"] = 76
    cfg1["_model:hidden_size_4"] = 76
    cfg1["_model:lr_log10"] = -3.75
    cfg1["_model:nonlintype"] = 'tanh'
    cfg1["_task_transformer_0:dx_log10Fgain"] = 2.25
    cfg1["_task_transformer_0:dx_log10Qgain"] = -1.25
    cfg1["_task_transformer_0:omega_log10Fgain"] = 2.25
    cfg1["_task_transformer_0:omega_log10Qgain"] = 2.25
    cfg1["_task_transformer_0:theta_log10Fgain"] = -1.25
    cfg1["_task_transformer_0:theta_log10Qgain"] = 2.25
    cfg1["_task_transformer_0:u_log10Rgain"] = -1.25
    cfg1["_task_transformer_0:x_log10Fgain"] = 2.25
    cfg1["_task_transformer_0:x_log10Qgain"] = -1.25

    cfg2 = pipeline.get_configuration_space().get_default_configuration()
    cfg2["_controller:horizon"] = 20
    cfg2["_model:n_hidden_layers"] = '1'
    cfg2["_model:hidden_size_1"] = 106
    cfg2["_model:lr_log10"] = -1.875
    cfg2["_model:nonlintype"] = 'selu'
    cfg2["_task_transformer_0:dx_log10Fgain"] = 3.125
    cfg2["_task_transformer_0:dx_log10Qgain"] = -2.125
    cfg2["_task_transformer_0:omega_log10Fgain"] = 1.375
    cfg2["_task_transformer_0:omega_log10Qgain"] = -2.125
    cfg2["_task_transformer_0:theta_log10Fgain"] = 3.125
    cfg2["_task_transformer_0:theta_log10Qgain"] = -0.375
    cfg2["_task_transformer_0:u_log10Rgain"] = 1.375
    cfg2["_task_transformer_0:x_log10Fgain"] = -2.125
    cfg2["_task_transformer_0:x_log10Qgain"] = -0.375

    cfg3 = pipeline.get_configuration_space().get_default_configuration()
    cfg3["_controller:horizon"] = 19
    cfg3["_model:n_hidden_layers"] = '1'
    cfg3["_model:hidden_size_1"] = 112
    cfg3["_model:lr_log10"] = -1.875
    cfg3["_model:nonlintype"] = 'sigmoid'
    cfg3["_task_transformer_0:dx_log10Fgain"] = 3.125
    cfg3["_task_transformer_0:dx_log10Qgain"] = -2.642851488370617
    cfg3["_task_transformer_0:omega_log10Fgain"] = 1.5757950604734452
    cfg3["_task_transformer_0:omega_log10Qgain"] = -2.3753935751317323
    cfg3["_task_transformer_0:theta_log10Fgain"] = 2.955399914571889
    cfg3["_task_transformer_0:theta_log10Qgain"] = -0.53825238466779
    cfg3["_task_transformer_0:u_log10Rgain"] = 1.3834653167227504
    cfg3["_task_transformer_0:x_log10Fgain"] = -2.125
    cfg3["_task_transformer_0:x_log10Qgain"] = -0.375

    cfg4 = pipeline.get_configuration_space().get_default_configuration()
    cfg4["_model:n_hidden_layers"] = '4'
    cfg4["_controller:horizon"] = 20
    cfg4["_model:hidden_size_1"] = 106
    cfg4["_model:hidden_size_2"] = 32
    cfg4["_model:hidden_size_3"] = 32
    cfg4["_model:hidden_size_4"] = 35
    cfg4["_model:lr_log10"] = -1.875
    cfg4["_model:nonlintype"] = 'tanh'
    cfg4["_task_transformer_0:dx_log10Fgain"] = 3.125
    cfg4["_task_transformer_0:dx_log10Qgain"] = -2.051623019774169
    cfg4["_task_transformer_0:omega_log10Fgain"] = 0.6428224090689199
    cfg4["_task_transformer_0:omega_log10Qgain"] = -2.2124535151247926
    cfg4["_task_transformer_0:theta_log10Fgain"] = 3.090361502171662
    cfg4["_task_transformer_0:theta_log10Qgain"] = -0.7742638703716658
    cfg4["_task_transformer_0:u_log10Rgain"] = 1.2412102069177653
    cfg4["_task_transformer_0:x_log10Fgain"] = -2.8981702020290703
    cfg4["_task_transformer_0:x_log10Qgain"] = -0.375

    cfg5 = pipeline.get_configuration_space().get_default_configuration()
    cfg5["_model:n_hidden_layers"] = '1'
    cfg5["_controller:horizon"] = 20
    cfg5["_model:hidden_size_1"] = 124
    cfg5["_model:lr_log10"] = -2.0624236642111304
    cfg5["_model:nonlintype"] = 'selu'
    cfg5["_task_transformer_0:dx_log10Fgain"] = 3.125
    cfg5["_task_transformer_0:dx_log10Qgain"] = -2.1647583484435087
    cfg5["_task_transformer_0:omega_log10Fgain"] = 1.2467268587695157
    cfg5["_task_transformer_0:omega_log10Qgain"] = -2.8025542923283036
    cfg5["_task_transformer_0:theta_log10Fgain"] = 3.5731179256547243
    cfg5["_task_transformer_0:theta_log10Qgain"] = -0.6494656550832953
    cfg5["_task_transformer_0:u_log10Rgain"] = 1.3834653167227504
    cfg5["_task_transformer_0:x_log10Fgain"] = -2.125
    cfg5["_task_transformer_0:x_log10Qgain"] = -0.12565529803933106

    cfg6 = pipeline.get_configuration_space().get_default_configuration()
    cfg6["_model:n_hidden_layers"] = '4'
    cfg6["_controller:horizon"] = 20
    cfg6["_model:hidden_size_1"] = 105
    cfg6["_model:hidden_size_2"] = 44
    cfg6["_model:hidden_size_3"] = 32
    cfg6["_model:hidden_size_4"] = 47
    cfg6["_model:lr_log10"] = -1.8613678376416387
    cfg6["_model:nonlintype"] = 'selu'
    cfg6["_task_transformer_0:dx_log10Fgain"] = 3.125
    cfg6["_task_transformer_0:dx_log10Qgain"] = -2.63439083089795
    cfg6["_task_transformer_0:omega_log10Fgain"] = 1.2467268587695157
    cfg6["_task_transformer_0:omega_log10Qgain"] = -2.1805539567995975
    cfg6["_task_transformer_0:theta_log10Fgain"] = 3.5572624796195216
    cfg6["_task_transformer_0:theta_log10Qgain"] = 0.07047112155650925
    cfg6["_task_transformer_0:u_log10Rgain"] = 1.3990807809309587
    cfg6["_task_transformer_0:x_log10Fgain"] = -2.644226465443066
    cfg6["_task_transformer_0:x_log10Qgain"] = -0.27751394660904793

    cfg7 = pipeline.get_configuration_space().get_default_configuration()
    cfg7["_model:n_hidden_layers"] = '1'
    cfg7["_controller:horizon"] = 21
    cfg7["_model:hidden_size_1"] = 78
    cfg7["_model:lr_log10"] = -1.9061089546626797
    cfg7["_model:nonlintype"] = 'selu'
    cfg7["_task_transformer_0:dx_log10Fgain"] = 3.227423052182771
    cfg7["_task_transformer_0:dx_log10Qgain"] = -2.321790382845302
    cfg7["_task_transformer_0:omega_log10Fgain"] = 1.315330715746514
    cfg7["_task_transformer_0:omega_log10Qgain"] = -2.6090383845598013
    cfg7["_task_transformer_0:theta_log10Fgain"] = 3.988295513101985
    cfg7["_task_transformer_0:theta_log10Qgain"] = -0.21952314435167253
    cfg7["_task_transformer_0:u_log10Rgain"] = 1.3429667304160553
    cfg7["_task_transformer_0:x_log10Fgain"] = -2.125
    cfg7["_task_transformer_0:x_log10Qgain"] = -0.24223917125694516

    cfg8 = pipeline.get_configuration_space().get_default_configuration()
    cfg8["_controller:horizon"] = 21
    cfg8["_model:n_hidden_layers"] = '4'
    cfg8["_model:hidden_size_1"] = 84
    cfg8["_model:hidden_size_2"] = 56
    cfg8["_model:hidden_size_3"] = 16
    cfg8["_model:hidden_size_4"] = 104
    cfg8["_model:lr_log10"] = -1.9033981276100183
    cfg8["_model:nonlintype"] = 'selu'
    cfg8["_task_transformer_0:dx_log10Fgain"] = 3.3696258925921327
    cfg8["_task_transformer_0:dx_log10Qgain"] = -2.194632487556923
    cfg8["_task_transformer_0:omega_log10Fgain"] = 1.3657341596284427
    cfg8["_task_transformer_0:omega_log10Qgain"] = -2.133358882125932
    cfg8["_task_transformer_0:theta_log10Fgain"] = 3.9922873251800253
    cfg8["_task_transformer_0:theta_log10Qgain"] = -0.4656027648595278
    cfg8["_task_transformer_0:u_log10Rgain"] = 2.4240761738900503
    cfg8["_task_transformer_0:x_log10Fgain"] = -2.125
    cfg8["_task_transformer_0:x_log10Qgain"] = 0.4572362741149525

    #cfgs = [cfg1, cfg2, cfg3, cfg3, cfg5, cfg6, cfg7]
    cfgs = [cfg8]

    surr_scoress = []
    true_scores = []
    for cfg in cfgs:
        torch.manual_seed(eval_seed)
        controller, model = pipeline(cfg, sysid_trajs)
        surr_scores = []
        _, true_score = eval_cfg(controller, surrogates[0])
        true_scores.append(true_score)
        for surrogate in surrogates:
            controller, _ = pipeline(cfg, sysid_trajs, model=model)
            surr_score = eval_cfg(controller, surrogate, eval_true=False)
            surr_scores.append(surr_score)
        surr_scoress.append(surr_scores)

    set_trace()

    return true_scores, surr_scoress
