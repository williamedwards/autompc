"""
Just the previous pipeline with nn as model and retraining on the dataset to update nn.
"""
import numpy as np
import torch

from .mppi import MPPI
from .. import zeros, extend


class MPPIAdaptive(MPPI):
    def __init__(self, network, cost_eqn, terminal_cost, model, **kwargs):
        """network is nn.Module that predicts next state given state and action in batch (tensor)
        cost_eqn evaluates path cost to be integrated (system sampling rate considered already) in batch mode (numpy)
        terminal_cost evaluates state terminal cost in batch mode (numpy)
        model is the autompc.Model instance used for forward simulation and data collection.
        """
        self.network = network
        dyn_fun = self.network_to_dyn(self.network)
        MPPI.__init__(self, dyn_fun, cost_eqn, terminal_cost, model, **kwargs)

    @staticmethod
    def network_to_dyn(network):
        """Convert a network into the dyn_eqn that accepts state, action matrix."""
        device = next(network.parameters()).device
        def dyn_eqn(state, action):
            network.eval()
            with torch.no_grad():
                netin = torch.from_numpy(np.concatenate((state, action), axis=1)).to(device)
                output = network(netin).cpu().numpy()
            return output
        return dyn_eqn

    @staticmethod
    def update_network(network, traj, **kwargs):
        """Given the collected trajectory, train the network to match the dynamics.
        It returns the training errors...
        """
        states = traj.obs[1:]
        ctrls = traj.ctrls[:-1]
        states_prev = traj.obs[:-1]
        n_epoch = kwargs.get('niter', 1)
        lr = kwargs.get('lr', 1e-3)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        loss = torch.nn.SmoothL1Loss()

        network.train()
        device = next(network.parameters()).device
        ts_state = torch.from_numpy(states).to(device)
        ts_input = torch.from_numpy(np.concatenate((states_prev, ctrls), axis=1)).to(device)
        errors = []
        for _ in range(n_epoch):
            optimizer.zero_grad()
            Yhat = network(ts_input)
            error = loss(Yhat, ts_state)
            errors.append(error.item())
            error.backward()
            optimizer.step()
        network.eval()
        print('error = ', errors[0], errors[-1])
        return errors

    def init_network(self, trajs, **config):
        """Just collect samples of data from model and train the network"""
        for traj in trajs:
            self.update_network(self.network, traj, **config)

    def run_episode(self, x0, nstep, update_config=None):
        """Just the run the system for one episode"""
        sim_traj = zeros(self.model.system, 1)
        sim_traj[0].obs[:] = x0
        # reinitialize the control sequence...
        self.act_sequence = self.noise_dist.sample((self.H,))
        for _ in range(nstep):
            u, _ = self.run(sim_traj)
            sim_traj.ctrls[-1] = u
            x = self.model.traj_to_state(sim_traj)
            x = self.model.pred(x, u)
            sim_traj = extend(sim_traj, [x], [np.zeros(self.model.system.ctrl_dim)])
        if update_config is not None and isinstance(update_config, dict):
            self.update_network(self.network, sim_traj, **update_config)
        return sim_traj