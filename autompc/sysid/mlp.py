"""
Just doing system id using multi-layer perceptron.
The code is similar to GP / RNN.
The configuration space has to be carefully considered
"""
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

from ..model import Model


class ForwardNet(torch.nn.Module):
    def __init__(self, n_in, n_out, n_hidden, hidden_size, nonlintype):
        """Specify the feedforward neuro network size and nonlinearity"""
        assert n_hidden > 0
        torch.nn.Module.__init__(self)
        self.layers = torch.nn.ModuleDict() # a collection that will hold your layers
        last_n = n_in
        for i in range(n_hidden):
            self.layers['layer%d' % i] = torch.nn.Linear(last_n, hidden_size)
            last_n = hidden_size
        # the final one
        self.output_layer = torch.nn.Linear(last_n, n_out)
        if nonlintype == 'relu':
            self.nonlin = torch.nn.ReLU()
        elif nonlintype == 'tanh':
            self.nonlin = torch.nn.Tanh()
        elif nonlintype == 'sigmoid':
            self.nonlin = torch.nn.Sigmoid()
        else:
            raise NotImplementedError("Currently supported nonlinearity: relu, tanh, sigmoid")

    def forward(self, x):
        for lyr in self.layers:
            x = self.nonlin(self.layers[lyr](x))
        return self.output_layer(x)


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]


class MLP(Model):
    def __init__(self, system, n_hidden=2, hidden_size=32, nonlintype='relu', n_iter=10, n_batch=64, lr=1e-3):
        Model.__init__(self, system)
        nx, nu = system.obs_dim, system.ctrl_dim
        self.net = ForwardNet(nx + nu, nx, n_hidden, hidden_size, nonlintype)
        self._train_data = (n_iter, n_batch, lr)
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.net = self.net.double().to(self._device)

    @staticmethod
    def get_configuration_space(system):
        cs = CS.ConfigurationSpace()
        method = CSH.CategoricalHyperparameter("method", choices=["lstsq", "lasso"])
        lasso_alpha_log10 = CSH.UniformFloatHyperparameter("lasso_alpha_log10", 
                lower=-5.0, upper=2.0, default_value=0.0)
        use_lasso_alpha = CSC.InCondition(child=lasso_alpha_log10, parent=method, 
                values=["lasso"])

        poly_basis = CSH.CategoricalHyperparameter("poly_basis", 
                choices=["true", "false"], default_value="false")
        poly_degree = CSH.UniformIntegerHyperparameter("poly_degree", lower=2, upper=8,
                default_value=3)
        use_poly_degree = CSC.InCondition(child=poly_degree, parent=poly_basis,
                values=["true"])

        trig_basis = CSH.CategoricalHyperparameter("trig_basis", 
                choices=["true", "false"], default_value="false")
        trig_freq = CSH.UniformIntegerHyperparameter("trig_freq", lower=1, upper=8,
                default_value=1)
        use_trig_freq = CSC.InCondition(child=trig_freq, parent=trig_basis,
                values=["true"])

        cs.add_hyperparameters([method, lasso_alpha_log10, poly_basis, poly_degree,
            trig_basis, trig_freq])
        cs.add_conditions([use_lasso_alpha, use_poly_degree, use_trig_freq])

        return cs

    def traj_to_state(self, traj):
        return traj[-1].obs.copy()
    
    def update_state(self, state, new_ctrl, new_obs):
        return new_obs.copy()

    @property
    def state_dim(self):
        return self.system.obs_dim

    def train(self, trajs):
        n_iter, n_batch, lr = self._train_data
        X = [traj.obs for traj in trajs]
        U = [traj.ctrls for traj in trajs]
        # concatenate data
        allXprev = np.concatenate([traj.obs[:-1] for traj in trajs], axis=0)
        allUprev = np.concatenate([traj.ctrls[:-1] for traj in trajs], axis=0)
        feedX = np.concatenate((allXprev, allUprev), axis=1)
        predY = np.concatenate([traj.obs[1:] for traj in trajs], axis=0)
        dataset = SimpleDataset(feedX, predY)
        dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=True)
        # now I can perform training... using torch default dataset holder
        self.net.train()
        for param in self.net.parameters():
            param.requires_grad_(True)
        optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        lossfun = torch.nn.SmoothL1Loss()
        for i in range(n_iter):
            print('Train iteration %d' % i)
            for x, y in dataloader:
                optim.zero_grad()
                x = x.to(self._device)
                predy = self.net(x)
                loss = lossfun(predy, y.to(self._device))
                loss.backward()
                optim.step()
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad_(False)

    def pred(self, state, ctrl):
        with torch.no_grad():
            xin = torch.from_numpy(np.concatenate((state, ctrl))[None, :]).to(self._device)
            yout = self.net(xin).cpu().numpy()[0]
        return yout

    def pred_parallel(self, state, ctrl):
        with torch.no_grad():
            xin = torch.from_numpy(np.concatenate((state, ctrl), axis=1)).to(self._device)
            yout = self.net(xin).cpu().numpy()
        return yout

    def pred_diff(self, state, ctrl):
        """Use code from https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa .
        
        def get_batch_jacobian(net, x, to):
            # noutputs: total output dim (e.g. net(x).shape(b,1,4,4) noutputs=1*4*4
            # b: batch
            # i: in_dim
            # o: out_dim
            # ti: total input dim
            # to: total output dim
            x_batch = x.shape[0]
            x_shape = x.shape[1:]
            x = x.unsqueeze(1)  # b, 1 ,i
            x = x.repeat(1, to, *(1,)*len(x.shape[2:]))  # b * to,i  copy to o dim
            x.requires_grad_(True)
            tmp_shape = x.shape
            y = net(x.reshape(-1, *tmp_shape[2:]))  # x.shape = b*to,i y.shape = b*to,to
            y_shape = y.shape[1:]  # y.shape = b*to,to
            y = y.reshape(x_batch, to, to)  # y.shape = b,to,to
            input_val = torch.eye(to).reshape(1, to, to).repeat(x_batch, 1, 1)  # input_val.shape = b,to,to  value is (eye)
            y.backward(input_val)  # y.shape = b,to,to
            return x.grad.reshape(x_batch, *y_shape, *x_shape).data  # x.shape = b,o,i
        """
        xin = torch.from_numpy(np.concatenate((state, ctrl))).to(self._device)
        n_out = self.system.obs_dim
        xin = xin.repeat(n_out, 1)
        xin.requires_grad_(True)
        yout = self.net(xin)
        # compute gradient...
        yout.backward(torch.eye(n_out).to(self._device))
        outy = yout[0].data.cpu().numpy()
        jac = xin.grad.data.cpu().numpy()
        return outy, jac[:, :n_out], jac[:, n_out:]

    @staticmethod
    def get_configuration_space(system):
        cs = CS.ConfigurationSpace()
        cs.add_configuration_space
        return cs