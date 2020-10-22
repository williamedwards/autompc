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

class ModelFromGym(ampc.Model):
    def __init__(self, tinf):
        super().__init__(tinf.system)
        self.env = gym.make(tinf.env_name)
        self.tinf = tinf

    def traj_to_state(self, traj):
        return traj[-1].obs[:]

    def update_state(self, state, new_ctrl, new_obs):
        return new_obs[:]

    def pred(obs, ctrl):
        self.tinf.set_env_state(self.env, state)
        action = self.tinf.ctrl_to_gym(ctrl)
        gym_obs, _, _, _ = selv.env.step(action)

    @staticmethod
    def get_configuration_space(system):
        raise NotImplementedError

    @property
    def state_dim(self):
        return self.system.obs_dim

def singlesim_surrogate(tinf, simsteps, sim_model):
    env = gym.make(tinf.env_name)
    def eval_cfg(pipeline, train_trajs, cfg):
        controller, model = pipeline(cfg, train_trajs)
        sim_traj = ampc.zeros(system, 1)
        x = np.copy(init_obs)
        sim_traj[0].obs[:] = x
        
        constate = controller.traj_to_state(sim_traj)
        simstate = sim_model.traj_to_state(sim_traj)
        cum_reward = 0
        for _  in range(self.sim_iters):
            tinf.set_env_state(env, simstate[:tinf.system.obs_dim])
            u, constate = controller.run(constate, sim_traj[-1].obs)
            _, reward, _, _ = env.step(tinf.ctrl_to_gym(u))
            cum_reward += reward
            simstate = sim_model.pred(simstate, u)
            x = simstate[:system.obs_dim]
            sim_traj[-1, "u"] = u
            sim_traj = extend(sim_traj, [x], [[0.0]])
        return cum_reward
    return eval_cfg

def truedyn_surrogate(tinf, simsteps):
    sim_model = ModelFromGym(tinf)
    return singlesim_surrogate(tinf, simsteps, sim_model)

def init_surrogate(tinf, simsteps, surrogate_name, surr_trajs):
    if surrogate_name == "truedyn":
        return truedyn_surrogate(tinf, simsteps)
