"""
Implement GP using GPytorch which naturally supports gradient computation.
It's fairly scalable since it uses GPU and some other tricks.
The gradient computation is a pain but eventually I was able to do it after some search.
"""
import copy
import tqdm
from pdb import set_trace

import numpy as np
import numpy.linalg as la

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

import torch
try:
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import VariationalStrategy
    from torch.utils.data import TensorDataset, DataLoader
except ImportError as e:
    print("GPytorch is not installed, cannot use GP model")
    raise e


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


class GPytorchGP(Model):
    """Define a base class that can be extended to both scalable and un-scalable case"""
    def __init__(self, system, mean='constant', kernel='RBF', niter=40, lr=0.1,
            use_cuda=True):
        super().__init__(system)
        self.niter = niter
        self.lr = lr
        self.device = (torch.device('cuda') if (use_cuda and torch.cuda.is_available()) 
                else torch.device('cpu'))
        if use_cuda and torch.cuda.is_available():
            print("Cuda is used for GPytorch")
        self.gpmodel = None
        self.gp_mean = mean
        self.gp_kernel = kernel

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

    def pred(self, state, ctrl):
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        # for this one, make a prediction is easy...
        TsrXt = torch.from_numpy(Xt).to(self.device)
        predy = self.gpmodel.likelihood(self.gpmodel(TsrXt))
        out = predy.mean.cpu().data.numpy()
        dy = transform_output(self.dy_means, self.dy_std, out).flatten()
        return state + dy

    def get_sampler(self):
        d = self.system.obs_dim
        u = np.random.normal(loc=0, scale=1, size=d).reshape((d, 1)) 
        def sample(state, ctrl):
            X = np.concatenate([state, ctrl])
            X = X[np.newaxis,:]
            Xt = transform_input(self.xu_means, self.xu_std, X)
            # for this one, make a prediction is easy...
            TsrXt = torch.from_numpy(Xt).to(self.device)
            predy = self.gpmodel.likelihood(self.gpmodel(TsrXt))
            #predf = self.gpmodel(TsrXt)
            mean = predy.mean.cpu().data.reshape((d,1))
            cov = predy.covariance_matrix.cpu().data
            L = np.linalg.cholesky(cov)
            out = mean + np.dot(L, u)
            out = out.reshape((1,d))
            #out2 = predy.sample().cpu().data.numpy()
            dy = transform_output(self.dy_means, self.dy_std, out).flatten()
            return state + dy
        return sample

    def sample(self, state, ctrl):
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        # for this one, make a prediction is easy...
        TsrXt = torch.from_numpy(Xt).to(self.device)
        predy = self.gpmodel.likelihood(self.gpmodel(TsrXt))
        #predf = self.gpmodel(TsrXt)
        d = self.system.obs_dim
        mean = predy.mean.cpu().data.reshape((d,1))
        cov = predy.covariance_matrix.cpu().data
        L = np.linalg.cholesky(cov)
        u = np.random.normal(loc=0, scale=1, size=d).reshape((d, 1)) 
        out = mean + np.dot(L, u)
        out = out.reshape((1,d))
        #out2 = predy.sample().cpu().data.numpy()
        dy = transform_output(self.dy_means, self.dy_std, out).flatten()
        return state + dy

    def pred_timeit(self, state, ctrl):
        import time
        start = time.time()
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Xt = transform_input(self.xu_means, self.xu_std, X)
        print("time1=", (time.time() - start)*1000, "ms")
        # for this one, make a prediction is easy...
        TsrXt = torch.from_numpy(Xt).to(self.device)
        print("time2=", (time.time() - start)*1000, "ms")
        predy = self.gpmodel.likelihood(self.gpmodel(TsrXt))
        print("time3=", (time.time() - start)*1000, "ms")
        out = predy.mean.cpu().data.numpy()
        print("time4=", (time.time() - start)*1000, "ms")
        dy = transform_output(self.dy_means, self.dy_std, out).flatten()
        print("time5=", (time.time() - start)*1000, "ms")
        return state + dy

    def pred_batch(self, state, ctrl):
        """The batch mode"""
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        TsrXt = torch.from_numpy(Xt).to(self.device)
        predy = self.gpmodel.likelihood(self.gpmodel(TsrXt))
        out = predy.mean.cpu().data.numpy()
        dy = transform_output(self.dy_means, self.dy_std, out).flatten()
        return state + dy.reshape((state.shape[0], self.state_dim))

    def sample_parallel(self, state, ctrl):
        """The batch mode"""
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        TsrXt = torch.from_numpy(Xt).to(self.device)
        predy = self.gpmodel.likelihood(self.gpmodel(TsrXt))
        out = predy.sample().cpu().data.numpy()
        dy = transform_output(self.dy_means, self.dy_std, out).flatten()
        return state + dy.reshape((state.shape[0], self.state_dim))

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
        predy.backward(torch.eye(obs_dim).to(self.device), retain_graph=True)
        jac = TsrXt.grad.cpu().data.numpy()
        # properly scale back...
        jac = jac / self.xu_std[None] * self.dy_std[:, np.newaxis]
        # since repeat, y value is the first one...
        out = predy[:1,:].cpu().data.numpy()
        dy = transform_output(self.dy_means, self.dy_std, out).flatten()
        n = self.system.obs_dim
        state_jac = jac[:, :n] + np.eye(n)
        ctrl_jac = jac[:, n:]
        return state + dy, state_jac, ctrl_jac

    def pred_diff_parallel(self, state, ctrl):
        """Prediction, but with gradient information"""
        X = np.concatenate([state, ctrl], axis=1)
        Xt = transform_input(self.xu_means, self.xu_std, X)
        obs_dim = state.shape[1]
        m = state.shape[0]
        # get the Tensor
        TsrXt = torch.from_numpy(Xt).to(self.device)
        TsrXt = TsrXt.repeat(obs_dim, 1, 1).permute(1,0,2).flatten(0,1)
        TsrXt.requires_grad_(True)
        predy = self.gpmodel.likelihood(self.gpmodel(TsrXt)).mean
        predy.backward(torch.eye(obs_dim).to(self.device).repeat(m,1), retain_graph=True)
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


    @property
    def state_dim(self):
        return self.system.obs_dim

    def get_parameters(self):
        raise NotImplementedError

    def set_parameters(self, params):
        raise NotImplementedError


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


class ApproximateGPytorchModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_task, mean='constant', kernel='RBF'):
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        size = torch.Size([num_task])
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=size
        )

        # We have to wrap the VariationalStrategy in a MultitaskVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ), num_tasks=num_task
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=size)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=size),
            batch_shape=size
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LargeGaussianProcess(GPytorchGP):
    def __init__(self, system, mean='constant', kernel='RBF', niter=40, lr=0.1):
        super().__init__(system, mean, kernel, niter, lr)
        self.gpmodel = BatchIndependentMultitaskGPModel(self.system.obs_dim, mean, kernel).double()
        self.gpmodel = self.gpmodel.to(self.device)

    def train(self, trajs, silent=False):
        # Initialize kernels
        self.gpmodel.train()
        self.gpmodel.likelihood.train()

        optimizer = torch.optim.Adam(self.gpmodel.parameters(), lr=self.lr)  # Includes GaussianLikelihood parameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gpmodel.likelihood, self.gpmodel)

        # prepare data
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

        # convert into desired tensor
        train_x = torch.from_numpy(XUt)
        train_y = torch.from_numpy(dYt)
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device).contiguous()
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


# this part implements the approximate GP
class ApproximateGPModelFactory(ModelFactory):
    """
    Gaussian Processes (GPs) are a non-parametric regression method.  Since GPs have trouble
    scaling to large training sets, this class provides a variational GP which automatically
    selects a subset of the training data for inference. This functionality is provided by
    the gPyTorch library. For more details see the original documentation_, and the
    corresponding paper at https://arxiv.org/pdf/1411.2005.pdf 

    .. _documentation: https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html 

    Hyperparameters:

    - *induce_count* (Type: int, Lower: 50, Upper: 200, Default: 100): Number of inducing points
      to include in the gaussian process. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Model = ApproximateGPModel
        self.name = "ApproximateGP"

    def get_configuration_space(self):
        cs = ConfigurationSpace()
        induce_count = UniformIntegerHyperparameter("induce_count", lower=50,
                upper=200, default_value=100)
        cs.add_hyperparameter(induce_count)
        return cs

class ApproximateGPModel(GPytorchGP, Model):
    def __init__(self, system, mean='constant', kernel='RBF', niter=5, lr=0.1, batch_size=1024, induce_count=500, **kwargs):
        super().__init__(system, mean, kernel, niter, lr, **kwargs)
        self.batch_size = batch_size
        self.induce_count = induce_count

    def train(self, trajs, silent=False):
        """Given collected trajectories, train the GP to approximate the actual dynamics"""
        # extract transfer pairs from data
        X = np.concatenate([traj.obs[:-1,:] for traj in trajs])
        dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
        num_task = dY.shape[1]
        self.num_task = num_task
        U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs])
        XU = np.concatenate((X, U), axis = 1) # stack X and U together
        self.xu_means = np.mean(XU, axis=0)
        self.xu_std = np.std(XU, axis=0)
        XUt = transform_input(self.xu_means, self.xu_std, XU)

        self.dy_means = np.mean(dY, axis=0)
        self.dy_std = np.std(dY, axis=0)
        dYt = transform_input(self.dy_means, self.dy_std, dY)

        # convert into desired tensor data loader
        train_x = torch.from_numpy(XUt)
        train_y = torch.from_numpy(dYt)
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # construct the approximate GP instance
        induce = torch.stack([train_x[:self.induce_count] for _ in range(num_task)], dim=0)
        self.induce = induce
        self.gpmodel = ApproximateGPytorchModel(induce, num_task, self.gp_mean, self.gp_kernel).double()
        self.gpmodel = self.gpmodel.to(self.device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_task)
        likelihood = likelihood.to(self.device)
        
        # Initialize kernels
        self.gpmodel.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.gpmodel.parameters()},
            {'params': likelihood.parameters()},
        ], lr=self.lr)

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(likelihood, self.gpmodel, num_data=train_y.size(0))

        if silent:
            itr = range(self.niter)
        else:
            itr = tqdm.tqdm(range(self.niter))
        for i in itr:
            # Within each iteration, we will go over each minibatch of data
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.gpmodel(x_batch)
                loss = -mll(output, y_batch)
                # minibatch_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()

        self.gpmodel.eval()
        likelihood.eval()
        self.gpmodel.likelihood = likelihood

    def get_parameters(self):
        return {"gpmodel_state" : self.gpmodel.state_dict(),
                "induce" : self.induce,
                "xu_means" : self.xu_means,
                "xu_std" : self.xu_std,
                "dy_means" : self.dy_means,
                "dy_std" : self.dy_std,
                "num_task" : self.num_task}

    def set_parameters(self, params):
        self.xu_means = params["xu_means"]
        self.xu_std = params["xu_std"]
        self.dy_means = params["dy_means"]
        self.dy_std = params["dy_std"]
        self.induce = params["induce"]
        self.num_task = params["num_task"]
        self.gpmodel = ApproximateGPModel(self.induce, self.num_task, self.gp_mean, 
                self.gp_kernel).double()
        self.gpmodel = self.gpmodel.to(self.device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.num_task)
        likelihood = likelihood.to(self.device)
        self.gpmodel.likelihood = likelihood
        self.gpmodel.load_state_dict(params["gpmodel_state"])
