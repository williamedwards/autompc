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

from pdb import set_trace

from ..model import Model

def transform_input(xu_means, xu_std, XU):
    XUt = []
    for i in range(XU.shape[1]):
        XUt.append((XU[:,i] - xu_means[i]) / xu_std[i])
    return np.vstack(XUt).T

def transform_output(xu_means, xu_std, XU):
    XUt = []
    for i in range(XU.shape[1]):
        XUt.append((XU[:,i] * xu_std[i]) + xu_means[i])
    return np.vstack(XUt).T

class ForwardNet(torch.nn.Module):
    def __init__(self, n_in, n_out, hidden_sizes, nonlintype):
        """Specify the feedforward neuro network size and nonlinearity"""
        assert len(hidden_sizes) > 0
        torch.nn.Module.__init__(self)
        self.layers = torch.nn.ModuleDict() # a collection that will hold your layers
        last_n = n_in
        for i, size in enumerate(hidden_sizes):
            self.layers['layer%d' % i] = torch.nn.Linear(last_n, size)
            last_n = size
        # the final one
        self.output_layer = torch.nn.Linear(last_n, n_out)
        if nonlintype == 'relu':
            self.nonlin = torch.nn.ReLU()
        elif nonlintype == 'selu':
            self.nonlin = torch.nn.SELU()
        elif nonlintype == 'tanh':
            self.nonlin = torch.nn.Tanh()
        elif nonlintype == 'sigmoid':
            self.nonlin = torch.nn.Sigmoid()
        else:
            raise NotImplementedError("Currently supported nonlinearity: relu, tanh, sigmoid")

    def forward(self, x):
        for i, lyr in enumerate(self.layers):
            y = self.layers[lyr](x)
            x = self.nonlin(y)
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
    def __init__(self, system, n_hidden_layers=3, hidden_size=128, 
            nonlintype='relu', n_train_iters=50, n_batch=64, lr_log10=-3,
            hidden_size_1=None, hidden_size_2=None, hidden_size_3=None,
            hidden_size_4=None,
            use_cuda=True):
        Model.__init__(self, system)
        lr = 10**lr_log10
        nx, nu = system.obs_dim, system.ctrl_dim
        n_hidden_layers = int(n_hidden_layers)
        hidden_sizes = [hidden_size] * n_hidden_layers
        print(f"use_cuda={use_cuda}")
        #print(f"{torch.cuda.is_available()=}")
        if use_cuda and torch.cuda.is_available():
            print("MLP Using Cuda")
        else:
            print("MLP Not Using Cuda")
        for i, size in enumerate([hidden_size_1, hidden_size_2, hidden_size_3,
                hidden_size_4]):
            if size is not None:
                hidden_sizes[i] = size
        self.net = ForwardNet(nx + nu, nx, hidden_sizes, nonlintype)
        self._train_data = (n_train_iters, n_batch, lr)
        self._device = (torch.device('cuda') if (use_cuda and torch.cuda.is_available()) 
                else torch.device('cpu'))
        self.net = self.net.double().to(self._device)

    @staticmethod
    def get_configuration_space(system):
        cs = CS.ConfigurationSpace()
        nonlintype = CSH.CategoricalHyperparameter("nonlintype", 
                choices=["relu", "tanh", "sigmoid", "selu"],
                default_value="relu")
                #choices=["relu"])
        n_hidden_layers = CSH.CategoricalHyperparameter("n_hidden_layers",
                choices=["1", "2", "3", "4"], default_value="2")
        hidden_size_1 = CSH.UniformIntegerHyperparameter("hidden_size_1",
                lower = 16, upper = 256, default_value=32)
        hidden_size_2 = CSH.UniformIntegerHyperparameter("hidden_size_2",
                lower = 16, upper = 256, default_value=32)
        hidden_size_3 = CSH.UniformIntegerHyperparameter("hidden_size_3",
                lower = 16, upper = 256, default_value=32)
        hidden_size_4 = CSH.UniformIntegerHyperparameter("hidden_size_4",
                lower = 16, upper = 256, default_value=32)
        hidden_cond_2 = CSC.InCondition(child=hidden_size_2, parent=n_hidden_layers,
                values=["2","3","4"])
        hidden_cond_3 = CSC.InCondition(child=hidden_size_3, parent=n_hidden_layers,
                values=["3","4"])
        hidden_cond_4 = CSC.InCondition(child=hidden_size_4, parent=n_hidden_layers,
                values=["4"])
        #n_train_iters = CSH.UniformIntegerHyperparameter("n_train_iters",
        #        lower = 10, upper = 100, default_value=20)
        lr_log10 = CSH.UniformFloatHyperparameter("lr_log10",
                lower = -5, upper = 0, default_value=-3)
        cs.add_hyperparameters([nonlintype, n_hidden_layers, hidden_size_1,
            hidden_size_2, hidden_size_3, hidden_size_4,
            lr_log10])
        cs.add_conditions([hidden_cond_2, hidden_cond_3, hidden_cond_4])
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
        X = np.concatenate([traj.obs[:-1,:] for traj in trajs])
        dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
        U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs])
        XU = np.concatenate((X, U), axis = 1) # stack X and U together
        self.xu_means = np.mean(XU, axis=0)
        self.xu_std = np.std(XU, axis=0)
        XUt = transform_input(self.xu_means, self.xu_std, XU)

        self.dy_means = np.mean(dY, axis=0)
        self.dy_std = np.std(dY, axis=0)
        dYt = transform_input(self.dy_means, self.dy_std, dY)
        # concatenate data
        feedX = XUt
        predY = dYt
        dataset = SimpleDataset(feedX, predY)
        dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=True)
        # now I can perform training... using torch default dataset holder
        self.net.train()
        for param in self.net.parameters():
            param.requires_grad_(True)
        optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        lossfun = torch.nn.SmoothL1Loss()
        best_loss = float("inf")
        best_params = None
        for i in range(n_iter):
            print('Train iteration %d' % i)
            cum_loss = 0.0
            for i, (x, y) in enumerate(dataloader):
                optim.zero_grad()
                x = x.to(self._device)
                predy = self.net(x)
                loss = lossfun(predy, y.to(self._device))
                loss.backward()
                cum_loss += loss.item()
                optim.step()
            print("loss=", cum_loss)
       #     if loss.item() < best_loss:
       #         best_loss = loss.item()
       #         best_params = self.net.state_dict()
        self.net.eval()
       # self.net.load_state_dict(best_params)
        for param in self.net.parameters():
            param.requires_grad_(False)
       # for i, (x, y) in enumerate(dataloader):
       #     x = x.to(self._device)
       #     predy = self.net(x)
       #     loss = lossfun(predy, y.to(self._device))
        print("final_loss=", loss.item())

    def pred(self, state, ctrl):
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        with torch.no_grad():
            xin = torch.from_numpy(Xt).to(self._device)
            yout = self.net(xin).cpu().numpy()
        dy = transform_output(self.dy_means, self.dy_std, yout).flatten()
        return state + dy

    def pred_parallel(self, state, ctrl):
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        with torch.no_grad():
            xin = torch.from_numpy(Xt).to(self._device)
            yout = self.net(xin).cpu().numpy()
        dy = transform_output(self.dy_means, self.dy_std, yout).flatten()
        return state + dy.reshape((state.shape[0], self.state_dim))

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
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        xin = torch.from_numpy(Xt).to(self._device)
        n_out = self.system.obs_dim
        xin = xin.repeat(n_out, 1)
        xin.requires_grad_(True)
        yout = self.net(xin)
        # compute gradient...
        yout.backward(torch.eye(n_out).to(self._device))
        jac = xin.grad.cpu().data.numpy()
        # properly scale back...
        jac = jac / self.xu_std[None] * self.dy_std[:, np.newaxis]
        n = self.system.obs_dim
        state_jac = jac[:, :n] + np.eye(n)
        ctrl_jac = jac[:, n:]
        out = yout.detach().cpu().numpy()[:1, :]
        dy = transform_output(self.dy_means, self.dy_std, out).flatten()
        return state+dy, state_jac, ctrl_jac

    def pred_diff_parallel(self, state, ctrl):
        """Prediction, but with gradient information"""
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        obs_dim = state.shape[1]
        m = state.shape[0]
        # get the Tensor
        TsrXt = torch.from_numpy(Xt).to(self._device)
        TsrXt = TsrXt.repeat(obs_dim, 1, 1).permute(1,0,2).flatten(0,1)
        TsrXt.requires_grad_(True)
        predy = self.net(TsrXt)
        predy.backward(torch.eye(obs_dim).to(self._device).repeat(m,1), retain_graph=True)
        predy = predy.reshape((m, obs_dim, obs_dim))
        #predy.backward(retain_graph=True)
        jac = TsrXt.grad.cpu().data.numpy()
        jac = jac.reshape((m, obs_dim, TsrXt.shape[-1]))
        # properly scale back...
        jac = jac / np.tile(self.xu_std, (m,obs_dim,1)) * np.tile(self.dy_std, (m,1))[:,:,np.newaxis]
        # since repeat, y value is the first one...
        out = predy[:,0,:].cpu().data.numpy()
        dy = transform_output(self.dy_means, self.dy_std, out)
        n = self.system.obs_dim
        state_jacs = jac[:, :, :n] + np.tile(np.eye(n), (m,1,1))
        ctrl_jacs = jac[:, :, n:]
        return state + dy, state_jacs, ctrl_jacs


    def get_parameters(self):
        return {"net_state" : self.net.state_dict(),
                "xu_means" : self.xu_means,
                "xu_std" : self.xu_std,
                "dy_means" : self.dy_means,
                "dy_std" : self.dy_std }


    def set_parameters(self, params):
        self.xu_means = params["xu_means"]
        self.xu_std = params["xu_std"]
        self.dy_means = params["dy_means"]
        self.dy_std = params["dy_std"]
        self.net.load_state_dict(params["net_state"])
