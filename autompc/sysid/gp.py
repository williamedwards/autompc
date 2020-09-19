# Created by William Edwards (wre2@illinois.edu)

from pdb import set_trace
import copy

import numpy as onp
import numpy.linalg as la

import jax
from jax.config import config; config.update("jax_enable_x64", True)
import jax.numpy as np

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from .gp_utils.kernels import RBF as RBF2
from .gp_utils.kernels import WhiteKernel as WhiteKernel2


from ..model import Model
#from ..hyper import IntRangeHyperparam

def gp_predict(gp, kern, X):
    if len(X.shape) < 2:
        X = X[:,jnp.newaxis]
    K_trans = kern(X, gp.X_train_)
    y_mean = K_trans.dot(gp.alpha_)  # Line 4 (y_mean = f_star)
    # undo normalisation
    y_mean = gp._y_train_std * y_mean + gp._y_train_mean
    return y_mean

def transform_input(xu_means, xu_std, XU):
    XUt = []
    for i in range(XU.shape[1]):
        XUt.append((XU[:,i] - xu_means[i]) / xu_std[i])
    return np.vstack(XUt).T

class GaussianProcess(Model):
    def __init__(self, system):
        super().__init__(system)

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
        kernel1 = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))

        rbf2 = RBF2(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
        white2 = WhiteKernel2(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1)) 
        kernel2 = 1.0 * rbf2 +  white2

        # Prepare data
        X = np.concatenate([traj.obs[:-1,:] for traj in trajs])
        Y = np.concatenate([traj.obs[1:,:] for traj in trajs])
        U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs])
        XU = np.concatenate((X, U), axis = 1) # stack X and U together
        self.xu_means = np.mean(XU, axis=0)
        self.xu_std = np.std(XU, axis=0)
        XUt = transform_input(self.xu_means, self.xu_std, XU)

        # Train gp
        self.gps = []
        self.kernels = []
        self.gp_predicts = []
        self.gp_jacs = []
        for i in range(Y.shape[1]):
            # Normalize
            y_train_mean = np.mean(Y[:,i])
            y_train_std = np.std(Y[:,i])
            norm_Y = (Y[:,i] - y_train_mean) / y_train_std
            gp = GaussianProcessRegressor(kernel=kernel1,
                    alpha=0.0).fit(XUt, norm_Y)
            gp._y_train_mean = y_train_mean
            gp._y_train_std = y_train_std
            kernel3 = copy.deepcopy(kernel2)
            kernel3.k1.k1.set_params(**gp.kernel_.k1.k1.get_params())
            kernel3.k1.k2.set_params(**gp.kernel_.k1.k2.get_params())
            kernel3.k2.set_params(**gp.kernel_.k2.get_params())
            self.gps.append(gp)
            self.kernels.append(kernel3)
            #self.gp_predict = jax.jit(lambda X: gp_predict(gp, kernel2, X))
            gp_pred = lambda X, kern=kernel3, gp=gp: gp_predict(gp, kern, X)
            self.gp_predicts.append(gp_pred)
            self.gp_jacs.append(jax.jacobian(gp_pred))

    def pred(self, state, ctrl):
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Y = []
        Y2 = []
        Xt = transform_input(self.xu_means, self.xu_std, X)
        for gp_predict in self.gp_predicts:
            Y.append(gp_predict(Xt)[0])
        return onp.array(Y)

    def pred_diff(self, state, ctrl):
        X = np.concatenate([state, ctrl])
        X = X[np.newaxis,:]
        Y = []
        Y2 = []
        Xt = transform_input(self.xu_means, self.xu_std, X)
        for gp_predict in self.gp_predicts:
            Y.append(gp_predict(Xt)[0])
        n = self.system.obs_dim
        m = self.system.ctrl_dim
        state_jac = onp.zeros((n,n))
        ctrl_jac = onp.zeros((n,m))
        for i, gp_jac in enumerate(self.gp_jacs):
            jac = gp_jac(Xt)
            state_jac[i, :n] = jac[0, 0, :n]
            ctrl_jac[i, n:] = jac[0, 0, n:]

        return onp.array(Y), state_jac, ctrl_jac

    @property
    def state_dim(self):
        return self.system.state_dim

    def get_parameters(self):
        return {"coeffs" : np.copy(self.coeffs)}

    def set_parameters(self, params):
        self.coeffs = np.copy(params["coeffs"])


