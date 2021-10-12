"""
Just doing system id using multi-layer perceptron.
The code is similar to GP / RNN.
The configuration space has to be carefully considered
"""
import itertools
import numpy as np
from tqdm import tqdm
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC
import copy

from pdb import set_trace

from .model import Model, ModelFactory

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


class MLPDADFactory(ModelFactory):
    """
    The multi-layer perceptron (MLP) model uses a feed-forward neural network
    architecutret to predict the system dynamics. The network size, activation
    function, and learning rate are tunable hyperparameters.

    Parameters

    - *n_batch* (Type: int, Default: 64): Training batch size of the neural net.
    - *n_train_iters* (Type: int, Default: 50): Number of training epochs

    Hyperparameters:

    - *n_hidden_layers* (Type: str, Choices: ["1", "2", "3", "4"], Default: "2"):
      The number of hidden layers in the network
    - *hidden_size_1* (Type int, Low: 16, High: 256, Default: 32): Size of hidden layer 1.
    - *hidden_size_2* (Type int, Low: 16, High: 256, Default: 32): Size of hidden layer 2. 
      (Conditioned on n_hidden_layers >=2).
    - *hidden_size_3* (Type int, Low: 16, High: 256, Default: 32): Size of hidden layer 3. 
      (Conditioned on n_hidden_layers >=3).
    - *hidden_size_4* (Type int, Low: 16, High: 256, Default: 32): Size of hidden layer 4. 
      (Conditioned on n_hidden_layers >=4).
    - *nonlintype* (Type: str, choices: ["relu", "tanh", "sigmoid", "selu"], Default: "relu):
      Type of activation function.
    - *lr* (Type: float, Low: 1e-5, High: 1, Default: 1e-3): Adam learning rate for the network.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Model = MLPDAD
        self.name = "MLPDAD"

    def get_configuration_space(self):
        cs = CS.ConfigurationSpace()
        nonlintype = CSH.CategoricalHyperparameter("nonlintype", 
                choices=["relu", "tanh", "sigmoid", "selu"],
                default_value="relu")
                #choices=["relu"])
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
        cs.add_hyperparameters([nonlintype, n_hidden_layers, hidden_size_1,
            hidden_size_2, hidden_size_3, hidden_size_4,
            lr])
        cs.add_conditions([hidden_cond_2, hidden_cond_3, hidden_cond_4])
        return cs

class MLPDAD(Model):
    def __init__(self, system, n_hidden_layers=3, hidden_size=128, 
            nonlintype='relu', n_train_iters=50, n_batch=64, lr=1e-3,
            hidden_size_1=None, hidden_size_2=None, hidden_size_3=None,
            hidden_size_4=None,
            use_cuda=True,
            n_dad_iters=2): # TODO: Find good default
        Model.__init__(self, system)
        nx, nu = system.obs_dim, system.ctrl_dim
        n_hidden_layers = int(n_hidden_layers)
        hidden_sizes = [hidden_size] * n_hidden_layers
        #print(f"{torch.cuda.is_available()=}")
        if use_cuda and torch.cuda.is_available():
            print("MLPDAD Using Cuda")
        else:
            if use_cuda:
                print("MLPDAD Not Using Cuda because torch.cuda is not available")
            else:
                print("MLPDAD Not Using Cuda")
        for i, size in enumerate([hidden_size_1, hidden_size_2, hidden_size_3,
                hidden_size_4]):
            if size is not None:
                hidden_sizes[i] = size
        print("hidden_sizes=", hidden_sizes)
        self.net = ForwardNet(nx + nu, nx, hidden_sizes, nonlintype)
        self._train_data = (n_train_iters, n_batch, lr, n_dad_iters)
        self._device = (torch.device('cuda') if (use_cuda and torch.cuda.is_available()) 
                else torch.device('cpu'))
        self.net = self.net.double().to(self._device)

    def traj_to_state(self, traj):
        return traj[-1].obs.copy()
    
    def update_state(self, state, new_ctrl, new_obs):
        return new_obs.copy()

    @property
    def state_dim(self):
        return self.system.obs_dim

    def train(self, trajs, silent=False, seed=100):
        torch.manual_seed(seed)
        n_iter, n_batch, lr, n_dad_iter = self._train_data

        originalNet = copy.deepcopy(self.net)

        # Initial Training of Model
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

        print("Training Initial MLP: ", end="\n")
        for i in tqdm(range(n_iter), file=sys.stdout):
            cum_loss = 0.0
            for i, (x, y) in enumerate(dataloader):
                optim.zero_grad()
                x = x.to(self._device)
                predy = self.net(x)
                loss = lossfun(predy, y.to(self._device))
                loss.backward()
                cum_loss += loss.item()
                optim.step()
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad_(False)


        # Train New Models and Add Data as Demonstrator
        best_net = copy.deepcopy(self.net)
        best_loss = cum_loss

        # trainedModels = [copy.deepcopy(self.net)]
        trainedModels = [self.net]
        modelsLoss = [cum_loss]

        print("\nTraining MLP with DAD: ", end="\n")
        for n in range(n_dad_iter):
            print("DaD Iteration", n, "of", n_dad_iter, end="\n")
            # Reset dataset to initial state
            # X = np.concatenate([traj.obs[:-1,:] for traj in trajs])
            # dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
            # U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs]) 
            print("Generating Predicted Trajectories: ", end="\n")
            for traj in tqdm(trajs, file=sys.stdout):
                predictedTrajectory = np.array([self.pred(traj[0].obs, traj[0].ctrl)]) # Initial Value at T = 1, xhat 1
                for t in range(1, traj.obs.shape[0] - 2): # Adding timesteps 2 through T - 1
                    predictedTrajectory = np.concatenate((predictedTrajectory, np.array([self.pred(predictedTrajectory[t - 1], traj[t].ctrl)])))

                # Adding feedX values
                X = np.concatenate((X, predictedTrajectory))

                # Adding delta Y values with obs(t) - pred(t - 1)
                sigma = traj.obs[2] # sigma 
                xhat = predictedTrajectory[0]
                difference = sigma - xhat
                newDY = np.array([difference])
                for t in range(1, predictedTrajectory.shape[0]):
                    sigma = traj.obs[t + 2]
                    xhat = predictedTrajectory[t]
                    difference = sigma - xhat
                    newDY = np.append(newDY, np.array([difference]), axis=0)
                
                dY = np.concatenate((dY, newDY))
                U = np.concatenate((U, traj.ctrls[1:-1]))
    
            XU = np.concatenate((X, U), axis = 1) # stack X and U together as X | U
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

            # train Nth model on untrained model
            self.net = copy.deepcopy(originalNet)
            self.net.train()
            for param in self.net.parameters():
                param.requires_grad_(True)
            optim = torch.optim.Adam(self.net.parameters(), lr=lr)
            lossfun = torch.nn.SmoothL1Loss()

            print("Training MLPDAD: ", end="\n")
            for i in tqdm(range(n_iter), file=sys.stdout):
                cum_loss = 0.0
                for i, (x, y) in enumerate(dataloader):
                    optim.zero_grad()
                    x = x.to(self._device)
                    predy = self.net(x)
                    loss = lossfun(predy, y.to(self._device))
                    loss.backward()
                    cum_loss += loss.item()
                    optim.step()
            self.net.eval()
            for param in self.net.parameters():
                param.requires_grad_(False)

            # TODO: Add evaluation for cumulative loss based on the original dataset, in the future consider hold out dataset
            # traj = trajs[0]
            # predictedTrajectory = np.array([traj[0].obs, traj[0].ctrl)]) # Initial Value at T = 1, T = 0 is redundant
            #     for t in range(1, traj.obs.shape[0] - 1): # Adding timesteps 2 through T
            #         predictedTrajectory = np.concatenate((predictedTrajectory, np.array([self.pred(predictedTrajectory[t - 1], traj[t].ctrl)])))

            #debugging models array that holds all previous models
            #trainedModels.append(copy.deepcopy(self.net))
            trainedModels.append(self.net)
            modelsLoss.append(cum_loss)


            if(cum_loss < best_loss): 
                best_net = copy.deepcopy(self.net)
                best_loss = cum_loss

        self.net = best_net
                    

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

    def pred_diff_batch(self, state, ctrl):
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
