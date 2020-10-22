# Created by William Edwards (wre@illinois.edu), 2020-10-21

# Standard library includes
import os, sys
from pdb import set_trace
import argparse

# External projects include
import numpy as np
import gym
import torch
from joblib import Memory
memory = Memory("cache")
#import gym_cartpole_swingup

# Internal project includes
import custom_gym_cartpole_swingup
import autompc as ampc
from autompc.sysid import ApproximateGaussianProcess, MLP
sys.path.append("../examples")
from surrogate_tuning_test import trajs2, cartpole

class ModelFromGym(ampc.Model):
    def __init__(self, tinf):
        super().__init__(tinf.system)
        self.env = gym.make(tinf.env_name)
        self.env.reset()
        self.tinf = tinf

    def traj_to_state(self, traj):
        return traj[-1].obs[:]

    def update_state(self, state, new_ctrl, new_obs):
        return new_obs[:]

    def pred(self, obs, ctrl):
        self.tinf.set_env_state(self.env, obs)
        action = self.tinf.ctrl_to_gym(ctrl)
        gym_obs, _, _, _ = self.env.step(action)
        return self.tinf.gym_to_obs(gym_obs)

    @staticmethod
    def get_configuration_space(system):
        raise NotImplementedError

    @property
    def state_dim(self):
        return self.system.obs_dim

@memory.cache
def sample_approx_gp_inner(num_trajs, seed):
    cs = ApproximateGaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, ApproximateGaussianProcess, cfg,
            use_cuda=False)
    #traj_sample = gen_trajs(traj_len=200, num_trajs=num_trajs, seed=seed)
    model.train(trajs2[-num_trajs:])
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

def runsim(tinf, simsteps, sim_model, controller):
    env = gym.make(tinf.env_name)
    sim_traj = ampc.zeros(tinf.system, 1)
    x = np.copy(tinf.init_obs)
    sim_traj[0].obs[:] = x
    
    constate = controller.traj_to_state(sim_traj)
    simstate = sim_model.traj_to_state(sim_traj)
    cum_reward = 0
    env.reset()
    for _  in range(simsteps):
        tinf.set_env_state(env, simstate[:tinf.system.obs_dim])
        u, constate = controller.run(constate, sim_traj[-1].obs)
        _, reward, _, _ = env.step(tinf.ctrl_to_gym(u))
        cum_reward += reward
        simstate = sim_model.pred(simstate, u)
        x = simstate[:tinf.system.obs_dim]
        print(f"{u=} {x=}")
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
    return cum_reward

def singlesim_surrogate(tinf, simsteps, sim_model):
    def eval_cfg(pipeline, train_trajs, cfg):
        torch.manual_seed(44)
        controller, model = pipeline(cfg, train_trajs)
        surrogate_cost = runsim(tinf, simsteps, sim_model, controller)
        truedyn_cost = runsim(tinf, simsteps, ModelFromGym(tinf), controller)
        return -surrogate_cost, truedyn_cost
    return eval_cfg

def truedyn_surrogate(tinf, simsteps):
    sim_model = ModelFromGym(tinf)
    return singlesim_surrogate(tinf, simsteps, sim_model)

def mlp_surrogate(tinf, simsteps):
    sim_model = sample_mlp(80, 0)
    return singlesim_surrogate(tinf, simsteps, sim_model)

def gp_surrogate(tinf, simsteps):
    sim_model = sample_approx_gp(80, 0)
    return singlesim_surrogate(tinf, simsteps, sim_model)

def init_surrogate(tinf, simsteps, surrogate_name, surr_trajs):
    if surrogate_name == "truedyn":
        return truedyn_surrogate(tinf, simsteps)
    elif surrogate_name == "mlp":
        return mlp_surrogate(tinf, simsteps)
    elif surrogate_name == "gp":
        return gp_surrogate(tinf, simsteps)
    else:
        raise
