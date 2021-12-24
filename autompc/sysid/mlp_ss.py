"""
Just doing system id using multi-layer perceptron.
The code is similar to GP / RNN.
The configuration space has to be carefully considered
"""
import itertools
import numpy as np
from numpy.core.numeric import roll
from tqdm import tqdm
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

import math
import matplotlib.pyplot as plt
#from autompc.benchmarks.cartpole import CartpoleSwingupBenchmark
from ..evaluation.model_metrics import get_model_rmse

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

class MLPSSFactory(ModelFactory):
    """
    Edited with Scheduled Sampling
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
        self.Model = MLPSS
        self.name = "MLPSS"

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
        end_epsilon = CSH.UniformFloatHyperparameter("endepsilon",
                lower = 0, upper = 1, default_value=0, log=False)
        start_epsilon = CSH.UniformFloatHyperparameter("startepsilon",
                lower = 0, upper = 1, default_value=1, log=False)
        cs.add_hyperparameters([nonlintype, n_hidden_layers, hidden_size_1,
            hidden_size_2, hidden_size_3, hidden_size_4,
            lr, end_epsilon, start_epsilon])
        cs.add_conditions([hidden_cond_2, hidden_cond_3, hidden_cond_4])
        return cs

class MLPSS(Model):
    def __init__(self, system, n_hidden_layers=3, hidden_size=128, 
            nonlintype='relu', n_train_iters=100, n_batch=64, lr=1e-3,
            hidden_size_1=None, hidden_size_2=None, hidden_size_3=None,
            hidden_size_4=None, seed=100,
            use_cuda=True,
            n_sampling_iters=100,
            endepsilon=0,
            startepsilon=1,
            test_trajectories=None):
        Model.__init__(self, system)
        nx, nu = system.obs_dim, system.ctrl_dim
        n_hidden_layers = int(n_hidden_layers)
        hidden_sizes = [hidden_size] * n_hidden_layers
        #print(f"{torch.cuda.is_available()=}")
        if use_cuda and torch.cuda.is_available():
            print("MLPSS Using Cuda")
        else:
            if use_cuda:
                print("MLPSSS Not Using Cuda because torch.cuda is not available")
            else:
                print("MLPSS Not Using Cuda")
        for i, size in enumerate([hidden_size_1, hidden_size_2, hidden_size_3,
                hidden_size_4]):
            if size is not None:
                hidden_sizes[i] = size
        print("hidden_sizes=", hidden_sizes)
        torch.manual_seed(seed)
        self.net = ForwardNet(nx + nu, nx, hidden_sizes, nonlintype)
        self._train_data = (n_train_iters, n_batch, lr)
        self._device = (torch.device('cuda') if (use_cuda and torch.cuda.is_available()) 
                else torch.device('cpu'))
        self.net = self.net.double().to(self._device)

        self.batchTrain = math.ceil(n_train_iters / (n_sampling_iters + 1))
        self.sampling_iters = n_sampling_iters
        self.testTrajectories = test_trajectories
        self.endEpsilon = endepsilon
        self.startEpsilon = startepsilon

        self.debug = False


    def traj_to_state(self, traj):
        return traj[-1].obs.copy()
    
    def update_state(self, state, new_ctrl, new_obs):
        return new_obs.copy()

    @property
    def state_dim(self):
        return self.system.obs_dim

    def trainScheduledIteration(self, dataloader):
        # now I can perform training... using torch default dataset holder
        n_iter, n_batch, lr = self._train_data
        self.net.train()
        for param in self.net.parameters():
            param.requires_grad_(True)
        optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        lossfun = torch.nn.SmoothL1Loss()
        best_loss = float("inf")
        best_params = None
        print("Training MLP: ", end="")
        for i in tqdm(range(self.batchTrain), file=sys.stdout):
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

    def train(self, trajs, silent=False, seed=100):
        torch.manual_seed(seed)
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

        # RMSE Dataset evaluation
        rmseTrainingError = []

        for i in range(self.sampling_iters + 1):
            X = np.empty((0,4))
            dY = np.empty((0,4))
            U = np.empty((0,1))
            iteration = i * self.batchTrain
            print("Training Model Iterations: ", iteration + 1, "-", iteration + self.batchTrain)
            print("Dataset Size: ", XU.shape[0])
            self.trainScheduledIteration(dataloader)

            if(self.debug):
                rmseTrainingError.append(get_model_rmse(self, self.testTrajectories, horizon=70))

            # Linear Sampling Function
            m = (self.startEpsilon - self.endEpsilon) / -n_iter
            epsilon = m * (iteration + self.batchTrain) + self.startEpsilon

            sampled = 0
            totalData = 0

            print("Generating Sampling Trajectories with epsilon of: ", epsilon)
            for traj in tqdm(trajs, file=sys.stdout):
                newTraj = np.copy(traj.obs)
                #rollout_states = traj[0].obs
                #for t in range(1, len(traj.ctrls) - 1):
                    #rollout_states = np.vstack((rollout_states,self.pred(rollout_states[t - 1,:],traj.ctrls[t - 1,:])))
                # Perform Sampling
                for t in range(1,len(newTraj)):
                    totalData = totalData + 1
                    rand = np.random.rand(1)[0]
                    if(rand >= epsilon):
                        sampled = sampled + 1
                        newTraj[t] = self.pred(newTraj[t - 1,:],traj.ctrls[t - 1,:])

                X = np.vstack((X, newTraj[:-1,:]))
                dY = np.vstack((dY, newTraj[1:,:] - newTraj[:-1,:]))
                U = np.vstack((U, traj.ctrls[:-1,:]))

                # Debug
                if(i == 1 and len(X) < 10000 and self.debug):
                    xAxisTimesteps = np.arange(len(newTraj))
                    variable = "Norm of Difference"
                    # plt.plot(xAxisTimesteps, newTraj[:,2].tolist(), label = variable + " Sampled at iter: " + str(iteration))
                    # plt.plot(xAxisTimesteps, traj.obs[:,2].tolist(), label = variable + " Observation at iter: " + str(iteration))
                    trajObs = traj.obs
                    plt.plot(xAxisTimesteps, np.linalg.norm(newTraj - traj.obs, axis=1).tolist(), label = variable + " Sampled at iter: " + str(iteration) + " e: " + str(epsilon))
                    #plt.plot(xAxisTimesteps, traj.obs[:,2].tolist(), label = variable + " Observation at iter: " + str(iteration))
                    plt.legend()
                    plt.savefig("Sampled Traj " + variable, dpi=600, bbox_inches='tight')
                    plt.clf()


            print("Percentage of Sampled Data: ", sampled / totalData)

            # Put together data
            XU = np.concatenate((X, U), axis = 1) # stack X and U together
            self.xu_means = np.mean(XU, axis = 0)
            self.xu_std = np.std(XU, axis = 0)
            XUt = transform_input(self.xu_means, self.xu_std, XU)

            self.dy_means = np.mean(dY, axis=0)
            self.dy_std = np.std(dY, axis=0)
            dYt = transform_input(self.dy_means, self.dy_std, dY)
            # concatenate data
            feedX = XUt
            predY = dYt
            dataset = SimpleDataset(feedX, predY)
            dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=True)

        if(self.debug):
            iterationsAxis = np.arange(n_iter + 1)
            plt.plot(iterationsAxis, rmseTrainingError, label = "RMSE Errors During Training")
            plt.legend()
            plt.savefig('RMSE Error During Training', dpi=600, bbox_inches='tight')
            plt.clf()

            testTraj = self.testTrajectories[0]
            xAxisTimesteps = np.arange(len(self.testTrajectories[0].obs))

            newTraj = np.copy(testTraj.obs)
            for t in range(1,len(newTraj)):
                totalData = totalData + 1
                if(np.random.rand(1)[0] >= epsilon):
                    sampled = sampled + 1
                    newTraj[t] = self.pred(newTraj[t - 1,:],testTraj.ctrls[t - 1,:])

            variable = "Theta"
            plt.plot(xAxisTimesteps, testTraj.obs[:,0].tolist(), label = variable + " Observation")
            plt.plot(xAxisTimesteps, newTraj[:,0].tolist(), label = variable + " Prediction: ")
            plt.legend()
            plt.savefig("TestTrajectory " + variable, dpi=600, bbox_inches='tight')
            plt.clf()
        #benchmark = CartpoleSwingupBenchmark()

        #testTraj = benchmark.gen_trajs(seed=300, n_trajs=1, traj_len=1000)[0]
        # testTraj = self.testTrajectory

        # newTraj = np.copy(testTraj.obs)
        # rollout_states = traj.obs
        # for t in range(1, len(traj.ctrls) - 1):
        #     rollout_states = np.vstack((rollout_states,self.pred(rollout_states[t - 1,:],traj.ctrls[t - 1,:])))

        # xAxisTimesteps = [0]
        # for t in range(1, testTraj[0].obs.shape[0]):
        #     xAxisTimesteps.append(t)

        # plt.plot(xAxisTimesteps, testTraj.obs[:,0].tolist(), label = "Theta Observation")
        # plt.plot(xAxisTimesteps, rollout_states[:,0].tolist(), label = "Theta Prediction: ")

        # plt.legend()
        # plt.savefig('trajTheta.png', dpi=600, bbox_inches='tight')
        # plt.clf()   
            

    def rolloutTraj(self, initialState, ctrls):
        #for t in range(ctrls.shape)
        return None    

    def pred(self, state, ctrl):
        X = np.concatenate((state, ctrl))
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
