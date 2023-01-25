
# Standard library includes
import sys
import time

# External library includes
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

# Internal library includes
from .model import Model,FullyObservableModel
from ..system import System
from ..trajectory import Trajectory

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
    def __init__(self, n_in, n_out, hidden_sizes, nonlintype, batchnorm=False):
        """Specify the feedforward neuro network size and nonlinearity"""
        assert len(hidden_sizes) > 0
        torch.nn.Module.__init__(self)
        self.layers = torch.nn.ModuleDict() # a collection that will hold your layers
        last_n = n_in
        for i, size in enumerate(hidden_sizes):
            layer = torch.nn.Linear(last_n, size)
            if batchnorm:
                layer = torch.nn.Sequential(layer,torch.nn.BatchNorm1d(size))
            self.layers['layer%d' % i] = layer
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
            raise NotImplementedError("Currently supported nonlinearity: relu, selu, tanh, sigmoid")

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


class MLP(FullyObservableModel):
    """
    The multi-layer perceptron (MLP) model uses a feed-forward neural network
    architecutret to predict the system dynamics. The network size, activation
    function, and learning rate are tunable hyperparameters.

    Parameters:

    - **n_batch** *(Type: int, Default: 64)*: Training batch size of the neural net.
    - **n_train_iters** *(Type: int, Default: 50)*: Number of training epochs
    - **use_cuda** *(Type: bool, Default: True)*: Use cuda if available.

    Hyperparameters:

    - **n_hidden_layers** *(Type: str, Choices: ["1", "2", "3", "4"], Default: "2")*:
      The number of hidden layers in the network
    - **hidden_size_1** *(Type int, Low: 16, High: 256, Default: 128)*: Size of hidden layer 1.
    - **hidden_size_2** *(Type int, Low: 16, High: 256, Default: 128)*: Size of hidden layer 2. 
      (Conditioned on n_hidden_layers >=2).
    - **hidden_size_3** *(Type int, Low: 16, High: 256, Default: 128)*: Size of hidden layer 3. 
      (Conditioned on n_hidden_layers >=3).
    - **hidden_size_4** *(Type int, Low: 16, High: 256, Default: 128)*: Size of hidden layer 4. 
      (Conditioned on n_hidden_layers >=4).
    - **nonlintype** *(Type: str, choices: ["relu", "tanh", "sigmoid", "selu"], Default: "relu)*:
      Type of activation function.
    - **lr** *(Type: float, Low: 1e-5, High: 1, Default: 1e-3)*: Adam learning rate for the network.
    """
    def __init__(self, system, n_train_iters=200, n_batch=64, use_cuda=True):
        super().__init__(system, "MLP")
        self.n_train_iters = n_train_iters
        self.n_batch = n_batch
        self.train_time_budget = None
        self.use_cuda = use_cuda
        self.net = None
        print(f"{use_cuda=}")
        print(f"{torch.cuda.is_available()=}")
        self._device = (torch.device('cuda') if (use_cuda and torch.cuda.is_available()) 
                else torch.device('cpu'))
        print(f"{self._device=}")

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        nonlintype = CSH.CategoricalHyperparameter("nonlintype", 
                choices=["relu", "tanh", "sigmoid", "selu"],
                default_value="relu")
        n_hidden_layers = CSH.CategoricalHyperparameter("n_hidden_layers",
                choices=["1", "2", "3", "4"], default_value="2")
        hidden_size_1 = CSH.UniformIntegerHyperparameter("hidden_size_1",
                lower = 16, upper = 256, default_value=128)
        hidden_size_2 = CSH.UniformIntegerHyperparameter("hidden_size_2",
                lower = 16, upper = 256, default_value=128)
        hidden_size_3 = CSH.UniformIntegerHyperparameter("hidden_size_3",
                lower = 16, upper = 256, default_value=128)
        hidden_size_4 = CSH.UniformIntegerHyperparameter("hidden_size_4",
                lower = 16, upper = 256, default_value=128)
        hidden_cond_2 = CSC.InCondition(child=hidden_size_2, parent=n_hidden_layers,
                values=["2","3","4"])
        hidden_cond_3 = CSC.InCondition(child=hidden_size_3, parent=n_hidden_layers,
                values=["3","4"])
        hidden_cond_4 = CSC.InCondition(child=hidden_size_4, parent=n_hidden_layers,
                values=["4"])
        lr = CSH.UniformFloatHyperparameter("lr",
                lower = 1e-5, upper = 1, default_value=1e-3, log=True)
        batchnorm = CSH.CategoricalHyperparameter("batchnorm",
                choices=[False,True], default_value=False)
        cs.add_hyperparameters([nonlintype, n_hidden_layers, batchnorm, hidden_size_1,
            hidden_size_2, hidden_size_3, hidden_size_4,
            lr])
        cs.add_conditions([hidden_cond_2, hidden_cond_3, hidden_cond_4])
        return cs

    def set_config(self, config):
        self.n_hidden_layers = int(config["n_hidden_layers"])
        self.nonlintype = config["nonlintype"]
        self.hidden_sizes = [config[f"hidden_size_{i}"] 
                             for i in range(1,self.n_hidden_layers+1)]
        self.lr = config["lr"]
        self.batchnorm = config["batchnorm"]

    def clear(self):
        self.net = None
        self.is_trained = False

    def set_device(self, device):
        self._device = device
        self.net = self.net.to(device)

    def _init_net(self, seed=100):
        torch.manual_seed(seed)
        self.net = ForwardNet(self.system.obs_dim + self.system.ctrl_dim, self.system.obs_dim, 
            self.hidden_sizes, self.nonlintype, self.batchnorm)
        self.net = self.net.double().to(self._device)
        print(f"{self.net.parameters().__next__().device=}")

    def _set_pairs(self, XU, dY):
        self.XU = XU
        self.dY = dY

    def _prepare_data(self):
        self.xu_means = np.mean(self.XU, axis=0)
        self.xu_std = np.std(self.XU, axis=0)
        XUt = transform_input(self.xu_means, self.xu_std, self.XU)
        self.dy_means = np.mean(self.dY, axis=0)
        self.dy_std = np.std(self.dY, axis=0)
        dYt = transform_input(self.dy_means, self.dy_std, self.dY)
        feedX = XUt
        predY = dYt
        dataset = SimpleDataset(feedX, predY)
        self.dataloader = DataLoader(dataset, batch_size=self.n_batch, shuffle=True)

    def _init_train(self, seed):
        self._init_net(seed)
        torch.manual_seed(seed)

        self.net.train()
        for param in self.net.parameters():
            param.requires_grad_(True)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.lossfun = torch.nn.SmoothL1Loss()

    def _step_train(self):
        cum_loss = 0.0
        for i, (x, y) in enumerate(self.dataloader):
            self.optim.zero_grad()
            x = x.to(self._device)
            predy = self.net(x)
            loss = self.lossfun(predy, y.to(self._device))
            loss.backward()
            cum_loss += loss.item()
            self.optim.step()

    def _finish_train(self):
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad_(False)
        self.is_trained = True
    
    def set_train_budget(self, seconds=None):
        self.train_time_budget = seconds

    def train(self, trajs, silent=False, seed=100):
        X = np.concatenate([traj.obs[:-1,:] for traj in trajs])
        dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
        U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs])
        XU = np.concatenate((X, U), axis = 1) # stack X and U together
        self._set_pairs(XU, dY)
        self._prepare_data()
        self._init_train(seed)

        print("Training MLP: ", end="")
        t0 = time.time()
        for i in tqdm(range(self.n_train_iters), file=sys.stdout):
            self._step_train()
            if self.train_time_budget is not None and time.time()-t0 > self.train_time_budget:
                print("Reached timeout of %.2fs"%self.train_time_budget)
                break

        self._finish_train()

    def pred(self, state, ctrl):
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        with torch.no_grad():
            xin = torch.from_numpy(Xt).to(self._device)
            yout = self.net(xin).cpu().numpy()
        dy = transform_output(self.dy_means, self.dy_std, yout).flatten()
        return state + dy

    def pred_batch(self, state, ctrl):
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        with torch.no_grad():
            xin = torch.from_numpy(Xt).to(self._device)
            yout = self.net(xin).cpu().numpy()
        dy = transform_output(self.dy_means, self.dy_std, yout).flatten()
        return state + dy.reshape((state.shape[0], self.state_dim))

    def pred_diff(self, state, ctrl):
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        xin = torch.from_numpy(Xt).to(self._device)
        n_out = self.system.obs_dim
        xin = xin.repeat(n_out, 1)
        xin.requires_grad_(True)
        yout = self.net(xin)
        # compute gradient...
        eye_val = torch.eye(n_out).to(self._device)
        eye_val = eye_val.type(yout.dtype)
        yout.backward(eye_val)
        jac = xin.grad.cpu().data.numpy()
        # properly scale back...
        jac = jac / self.xu_std[None] * self.dy_std[:, np.newaxis]
        n = self.system.obs_dim
        state_jac = jac[:, :n] + np.eye(n)
        ctrl_jac = jac[:, n:]
        out = yout.detach().cpu().numpy()[:1, :]
        dy = transform_output(self.dy_means, self.dy_std, out).flatten()
        return state+dy, state_jac, ctrl_jac

    def pred_diff_batch(self, state, ctrl):
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        obs_dim = state.shape[1]
        m = state.shape[0]
        # get the Tensor
        TsrXt = torch.from_numpy(Xt).to(self._device)
        TsrXt = TsrXt.repeat(obs_dim, 1, 1).permute(1,0,2).flatten(0,1)
        TsrXt.requires_grad_(True)
        predy = self.net(TsrXt)
        eye_val = torch.eye(obs_dim).to(self._device).repeat(m,1)
        eye_val = eye_val.type(predy.dtype)
        predy.backward(eye_val, retain_graph=True)
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
        if self.net is None:
            self._init_net()
        self.net.load_state_dict(params["net_state"])

class ARMLP(MLP):
    def __init__(self, system, history=4, n_train_iters=200, n_batch=64, use_cuda=True):
        super().__init__(system, n_train_iters, n_batch, use_cuda)
        self.k = history

        # NOTE THIS CLASS INHERITS FROM FULLY OBSERVABLE MODEL AND HENCE ASSUMES SO
        # # shift out old states
        # state[...,self.system.state_dim:self.system.state_dim*self.k] = state[:,:self.system.state_dim*(self.k-1)]
        # # shift out old control
        # state[...,self.system.state_dim*self.k+self.system.ctrl_dim:] = state[:,self.system.state_dim*self.k:self.system.state_dim*self.k+self.system.ctrl_dim]
        # # shift in new control
        # state[...,self.system.state_dim*self.k:self.system.state_dim*self.k+self.system.ctrl_dim] = new_ctrl

        # First we construct the system matrices
        A = np.zeros((self.state_dim, self.state_dim))
        B = np.zeros((self.state_dim, self.system.ctrl_dim))

        # Shift history
        n = self.system.obs_dim
        l = self.system.ctrl_dim
        m = self.system.obs_dim + self.system.ctrl_dim
        k = self.k

        A[0 : n, 0 : n] = np.eye(n) # NN outputs dY
        if k > 1:    
            A[n : 2*n, 0 : n] = np.eye(n)
        for i in range(k-2):
            A[(i+1)*m+n : (i+2)*m+n, i*m+n : (i+1)*m+n] = np.eye(m)

        # Predict new observation
        # A[0 : n, :] = coeffs[:, :-l]

        # Add new control
        # B[0 : n, :] = coeffs[:, -l:]
        B[2*n : 2*n + l, :] = np.eye(l)

        self.A, self.B = A, B

    @property
    def state_dim(self):
        return self._get_fvec_size() - self.system.ctrl_dim
    """
    Augmenting state space from 
    [x[t], u[t]] to [x[t],x[t-1],u[t-1],...,x[t-k+1],u[t-k+1], u[t]]
    """
    @property
    def state_system(self):
        vars = []
        for i in range(self.k):
            for x in self.system.observations:
                if i==0:
                    vars.append(x + '[t]')
                else:
                    vars.append(x + '[t-%d]'%i)
            for u in self.system.controls:
                if i==0 or i+1==self.k:
                    break
                else:
                    vars.append(u + '[t-%d]'%i)
        for u in self.system.controls:
            vars.append(u + '[t]')
        assert len(vars)==self.state_dim
        return System(vars,self.system.controls)

    def _get_feature_vector(self, traj, t=None):
        k = self.k
        if t is None:
            t = len(traj)

        feature_elements = [traj[max(t-1,0)].obs]
        for i in range(t-2, t-k-1, -1):
            if i < 0:
                feature_elements += [traj[0].obs, np.zeros(self.system.ctrl_dim)]
            else:
                feature_elements += [traj[i].obs, traj[i].ctrl]
            #feature_elements += [traj[max(i,0)].obs, traj[max(i,0)].ctrl]
        feature_elements += [traj[max(t-1,0)].ctrl]
        return np.concatenate(feature_elements)
    
    def _get_all_feature_vectors(self, traj):
        k = self.k
        feature_vectors = np.zeros((len(traj), k*(self.system.obs_dim + self.system.ctrl_dim)))
        feature_vectors[:,:self.system.obs_dim] = traj.obs
        j = self.system.obs_dim
        for i in range(1, k, 1):
            feature_vectors[:,j:j+self.system.obs_dim] = np.concatenate([traj.obs[:1,:]]*i + [traj.obs[:-i, :]])
            j += self.system.obs_dim
            feature_vectors[:,j:j+self.system.ctrl_dim] = np.concatenate([np.zeros_like(traj.ctrls[:1,:])]*i + [traj.ctrls[:-i, :]])
            #feature_vectors[:,j:j+self.system.ctrl_dim] = np.concatenate([traj.ctrls[:1,:]]*i + [traj.ctrls[:-i, :]])
            j += self.system.ctrl_dim
        feature_vectors[:,-self.system.ctrl_dim:] = traj.ctrls
        return feature_vectors

    def _get_fvec_size(self):
        k = self.k
        return k*self.system.obs_dim + k*self.system.ctrl_dim

    def _init_net(self, seed=100):
        torch.manual_seed(seed)
        self.net = ForwardNet(self._get_fvec_size(), self.system.obs_dim, 
            self.hidden_sizes, self.nonlintype, self.batchnorm)
        self.net = self.net.double().to(self._device)
        print(f"{self.net.parameters().__next__().device=}")

    def train(self, trajs, silent=False, seed=100):
        XU = np.concatenate([self._get_all_feature_vectors(traj[:-1]) for traj in trajs])
        dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
        assert XU.shape[1]==self._get_fvec_size()
        self._set_pairs(XU, dY)
        self._prepare_data()
        self._init_train(seed)

        print("Training MLP: ", end="")
        t0 = time.time()
        for i in tqdm(range(self.n_train_iters), file=sys.stdout):
            self._step_train()
            if self.train_time_budget is not None and time.time()-t0 > self.train_time_budget:
                print("Reached timeout of %.2fs"%self.train_time_budget)
                break

        self._finish_train()
    
    def update_state(self, state, new_ctrl, new_obs):
        # Shift the targets
        newstate = self.A @ state + self.B @ new_ctrl
        newstate[:self.system.obs_dim] = new_obs

        return newstate
    
    def traj_to_state(self, traj):
        return self._get_feature_vector(traj)[:-self.system.ctrl_dim]

    def init_state(self, obs : np.ndarray) -> np.ndarray:
        traj = Trajectory.zeros(self.system, 1)
        traj[0].obs[:] = obs
        return self.traj_to_state(traj)

    def get_obs(self, state):
        return state[0:self.system.obs_dim]

    def traj_to_states(self, traj):
        return self._get_all_feature_vectors(traj)[:, :-self.system.ctrl_dim]


    def pred(self, state, ctrl):
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        with torch.no_grad():
            xin = torch.from_numpy(Xt).to(self._device)
            yout = self.net(xin).cpu().numpy()
        # dy = np.zeros(state.shape)
        # dy[:self.system.obs_dim] = transform_output(self.dy_means, self.dy_std, yout).flatten()
        # return self.shift_state(state, ctrl) + dy
        dy = transform_output(self.dy_means, self.dy_std, yout).flatten()
        obs = state[:self.system.obs_dim]
        return self.update_state(state, ctrl, obs+dy)

    def pred_batch(self, state, ctrl):
        # Predict next state        
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        with torch.no_grad():
            xin = torch.from_numpy(Xt).to(self._device)
            yout = self.net(xin).cpu().numpy()
        # dy = np.zeros(state.shape)
        # dy[:,:self.system.obs_dim] = transform_output(self.dy_means, self.dy_std, yout)
        # return self.shift_state(state, ctrl) + dy
        dy = transform_output(self.dy_means, self.dy_std, yout)
        obs = state[:,:self.system.obs_dim]
        return self.update_state(state.T, ctrl.T, (obs+dy).T).T

    def pred_diff(self, state, ctrl):
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        xin = torch.from_numpy(Xt).to(self._device)
        n_out = self.system.obs_dim
        xin = xin.repeat(n_out, 1)
        xin.requires_grad_(True)
        yout = self.net(xin)
        # compute gradient...
        eye_val = torch.eye(n_out).to(self._device)
        eye_val = eye_val.type(yout.dtype)
        yout.backward(eye_val)
        jac = xin.grad.cpu().data.numpy()
        # properly scale back...
        jac = jac / self.xu_std[None] * self.dy_std[:, np.newaxis]
        n = self.system.obs_dim
        l = self.system.ctrl_dim
        state_jac = np.zeros_like(self.A)
        ctrl_jac = np.zeros_like(self.B)
        # print('DEBUG: jac', jac)
        # print('DEBUG: state_jac and jac shape', state_jac.shape, jac.shape)
        state_jac[:n,:self.state_system.obs_dim] = jac[:, :-l] 
        ctrl_jac[:n,:l] = jac[:, -l:] 
        out = yout.detach().cpu().numpy()[:1, :]
        dy = transform_output(self.dy_means, self.dy_std, out).flatten()
        obs = state[:n]
        state_jac += self.A
        ctrl_jac += self.B
        return self.update_state(state, ctrl, obs+dy), state_jac, ctrl_jac

    def pred_diff_batch(self, state, ctrl):
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        obs_dim = self.system.obs_dim
        m = state.shape[0]
        # get the Tensor
        TsrXt = torch.from_numpy(Xt).to(self._device)
        TsrXt = TsrXt.repeat(obs_dim, 1, 1).permute(1,0,2).flatten(0,1)
        TsrXt.requires_grad_(True)
        predy = self.net(TsrXt)
        eye_val = torch.eye(obs_dim).to(self._device).repeat(m,1)
        eye_val = eye_val.type(predy.dtype)
        predy.backward(eye_val, retain_graph=True)
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
        l = self.system.ctrl_dim
        state_jac = np.tile(np.zeros_like(self.A), (m,1,1))
        ctrl_jac = np.tile(np.zeros_like(self.B), (m,1,1))

        state_jac[:,:n,:self.state_system.obs_dim] = jac[:, :, :-l] 
        ctrl_jac[:,:n,:l] = jac[:, :, -l:] 
        obs = state[:,:n]
        state_jac += np.tile(self.A, (m,1,1))
        ctrl_jac += np.tile(self.B, (m,1,1))
        return self.update_state(state.T, ctrl.T, (obs+dy).T).T, state_jac, ctrl_jac