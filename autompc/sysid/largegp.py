# Standard libary includes
import time

# External library includes
import numpy as np
import numpy.linalg as la
import tqdm
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter

try:
    import torch
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import CholeskyVariationalDistribution
    from gpytorch.variational import VariationalStrategy
    from torch.utils.data import TensorDataset, DataLoader
except:
    from ..utils.exceptions import OptionalDependencyException
    raise OptionalDependencyException("GPytorch is not installed, cannot import this module")

# Internal library includes
from .model import Model,FullyObservableModel

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

class ApproximateGPytorchModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_task):
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

class ApproximateGPModel(FullyObservableModel):
    """
    Gaussian Processes (GPs) are a non-parametric regression method.  Since GPs have trouble
    scaling to large training sets, this class provides a variational GP which automatically
    selects a subset of the training data for inference. This functionality is provided by
    the gPyTorch library. For more details see the original documentation_, and the
    corresponding paper at https://arxiv.org/pdf/1411.2005.pdf 

    .. _documentation: https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html 

    Parameters:

    - **n_train_iters** *(Type: int, Default: 20)*: Number of training iterations
    - **batch_size** *(Type: int, Default: 1024)*: Batch size for training.

    Hyperparameters:

    - **induce_count** *(Type: int, Lower: 50, Upper: 200, Default: 100)*: Number of inducing points
      to include in the gaussian process. 
    - **learning_rate** *(Type: float, Lower: 1e-5, Upper: 10, Default: 0.1)*: Learning rate for training.
    """
    def __init__(self, system, n_train_iters=20, batch_size=1024, use_cuda=True):
        super().__init__(system, "ApproximateGPModel")
        self.batch_size = batch_size
        self.n_train_iters = n_train_iters
        self.train_time_budget = None
        self.device = (torch.device('cuda') if (use_cuda and torch.cuda.is_available()) 
                else torch.device('cpu'))
        if use_cuda and torch.cuda.is_available():
            print("Cuda is used for GPytorch")
        self.gpmodel = None

    def get_default_config_space(self):
        cs = ConfigurationSpace()
        induce_count = UniformIntegerHyperparameter("induce_count", lower=50,
                upper=200, default_value=100)
        lr = UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=10, 
                default_value=1e-1)
        cs.add_hyperparameters([induce_count,lr])
        return cs

    def set_config(self, config):
        self.induce_count = config["induce_count"]
        self.lr = config["learning_rate"]

    def clear(self):
        self.gpmodel = None
        self.is_trained = False

    def _set_pairs(self, XU, dY):
        self.XU = XU
        self.dY = dY
        
    def _prepare_data(self):
        self.num_task = self.dY.shape[1]
        self.xu_means = np.mean(self.XU, axis=0)
        self.xu_std = np.std(self.XU, axis=0)
        XUt = transform_input(self.xu_means, self.xu_std, self.XU)

        self.dy_means = np.mean(self.dY, axis=0)
        self.dy_std = np.std(self.dY, axis=0)
        dYt = transform_input(self.dy_means, self.dy_std, self.dY)

        # convert into desired tensor data loader
        train_x = torch.from_numpy(XUt)
        train_y = torch.from_numpy(dYt)
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        self.n_datapoints = train_y.shape[0]
        train_dataset = TensorDataset(train_x, train_y)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.induce = torch.stack([train_x[:self.induce_count] for _ in range(self.num_task)], dim=0)

    def _init_train(self, seed):
        torch.manual_seed(seed)
        self.gpmodel = ApproximateGPytorchModel(self.induce, self.num_task).double()
        self.gpmodel = self.gpmodel.to(self.device)

        # construct the approximate GP instance
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_task)
        self.likelihood = self.likelihood.to(self.device)

        # Initialize kernels
        self.gpmodel.train()
        self.likelihood.train()

        self.optimizer = torch.optim.Adam([
            {'params': self.gpmodel.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.lr)

        # Our loss object. We're using the VariationalELBO
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gpmodel, num_data=self.n_datapoints)

    def _step_train(self):
        for x_batch, y_batch in self.train_loader:
            self.optimizer.zero_grad()
            output = self.gpmodel(x_batch)
            loss = -self.mll(output, y_batch)
            # minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            self.optimizer.step()

    def _finish_train(self):
        self.gpmodel.eval()
        self.likelihood.eval()
        self.gpmodel.likelihood = self.likelihood
        self.is_trained = True

    def set_train_budget(self, seconds=None):
        self.train_time_budget = seconds

    def train(self, trajs, silent=False, seed=100):
        """Given collected trajectories, train the GP to approximate the actual dynamics"""
        # extract transfer pairs from data
        X = np.concatenate([traj.obs[:-1,:] for traj in trajs])
        dY = np.concatenate([traj.obs[1:,:] - traj.obs[:-1,:] for traj in trajs])
        U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs])
        XU = np.concatenate((X, U), axis = 1) # stack X and U together
        self._set_pairs(XU, dY)
        self._prepare_data()
        self._init_train(seed)

        if silent:
            itr = range(self.n_train_iters)
        else:
            itr = tqdm.tqdm(range(self.n_train_iters))
        t0 = time.time()
        for i in itr:
            self._step_train()
            if self.train_time_budget is not None and time.time()-t0 > self.train_time_budget:
                break

        self._finish_train()

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
        self.gpmodel = ApproximateGPytorchModel(self.induce, self.num_task).double()
        self.gpmodel = self.gpmodel.to(self.device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=self.num_task)
        likelihood = likelihood.to(self.device)
        self.gpmodel.likelihood = likelihood
        self.gpmodel.load_state_dict(params["gpmodel_state"])


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

