"""
Implement GP using GPytorch which naturally supports gradient computation.
It's fairly scalable since it uses GPU and some other tricks.
The gradient computation is a pain but eventually I was able to do it after some search.
"""
import copy

import numpy as np
import numpy.linalg as la

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

try:
    import torch
    import gpytorch
except:
    print("GPytorch is not installed, cannot import this module")


from ..model import Model


def transform_input(xu_means, xu_std, XU):
    XUt = []
    for i in range(XU.shape[1]):
        XUt.append((XU[:,i] - xu_means[i]) / xu_std[i])
    return np.vstack(XUt).T


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, num_task, mean='constant', kernel='RBF'):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_task)
        super().__init__(None, None, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_task]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_task])),
            batch_shape=torch.Size([num_task])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x), task_dim=0
        )


class LargeGaussianProcess(Model):
    def __init__(self, system, mean='constant', kernel='RBF', niter=100, lr=0.1):
        super().__init__(system)
        self.gpmodel = BatchIndependentMultitaskGPModel(self.system.obs_dim, mean, kernel).double()
        self.niter = niter
        self.lr = lr
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.gpmodel = self.gpmodel.to(self.device)

    @staticmethod
    def get_configuration_space(system):
        cs = ConfigurationSpace()
        return cs

    def update_state(self, state, new_ctrl, new_obs):
        return np.copy(new_obs)

    def traj_to_state(self, traj):
        return traj[-1].obs[:]

    def state_to_obs(self, state):
        return state[:]

    def train(self, trajs):
        # Initialize kernels
        self.gpmodel.train()
        self.gpmodel.likelihood.train()

        optimizer = torch.optim.Adam(self.gpmodel.parameters(), lr=self.lr)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gpmodel.likelihood, self.gpmodel)

        # prepare data
        X = np.concatenate([traj.obs[:-1,:] for traj in trajs])
        Y = np.concatenate([traj.obs[1:,:] for traj in trajs])
        U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs])
        XU = np.concatenate((X, U), axis = 1) # stack X and U together
        self.xu_means = np.mean(XU, axis=0)
        self.xu_std = np.std(XU, axis=0)
        XUt = transform_input(self.xu_means, self.xu_std, XU)

        # convert into desired tensor
        train_x = torch.from_numpy(XUt)
        train_y = torch.from_numpy(Y)
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        self.gpmodel.set_train_data(train_x, train_y, False)

        for i in range(self.niter):
            optimizer.zero_grad()
            output = self.gpmodel(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, self.niter, loss.item()))
            optimizer.step()
        # training is finished, now go to eval mode
        self.gpmodel.eval()
        self.gpmodel.likelihood.eval()

    def pred(self, state, ctrl):
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        # for this one, make a prediction is easy...
        TsrXt = torch.from_numpy(Xt).to(self.device)
        predy = self.gpmodel.likelihood(self.gpmodel(TsrXt))
        y = predy.mean.cpu().data.numpy()
        return y

    def pred_parallel(self, state, ctrl):
        """The batch mode"""
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        TsrXt = torch.from_numpy(Xt).to(self.device)
        predy = self.gpmodel.likelihood(self.gpmodel(TsrXt))
        y = predy.mean.cpu().data.numpy()  # TODO: check shape
        return y

    def pred_diff(self, state, ctrl):
        """Prediction, but with gradient information"""
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        obs_dim = len(state)
        # get the Tensor
        TsrXt = torch.from_numpy(Xt).to(self.device)
        TsrXt = TsrXt.repeat(obs_dim, 1)
        TsrXt.requires_grad_(True)
        predy = self.gpmodel.likelihood(self.gpmodel(TsrXt)).mean
        predy.backward(torch.eye(obs_dim).to(self.device))
        jac = TsrXt.grad.cpu().data.numpy()
        # properly scale back...
        jac = jac * self.xu_std[None]  # a row one for broadcasting
        # since repeat, y value is the first one...
        y = predy[0].cpu().data.numpy()
        n = self.system.obs_dim
        state_jac = jac[:, :n]
        ctrl_jac = jac[:, n:]
        return y, state_jac, ctrl_jac

    @property
    def state_dim(self):
        return self.system.state_dim

    def get_parameters(self):
        return {"coeffs" : np.copy(self.coeffs)}

    def set_parameters(self, params):
        self.coeffs = np.copy(params["coeffs"])