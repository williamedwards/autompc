

# Standard library includes
import pickle

# External library includes
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC
import pysindy as ps
import pysindy.differentiation as psd
from sklearn.linear_model import  Lasso

# Internal library includes
from .model import Model,FullyObservableModel
from .basis_funcs import *
from ..utils.cs_utils import (get_hyper_bool, get_hyper_str,
        get_hyper_float, get_hyper_int)

#the number of iters will be determined by this constant * budget
STLSQ_ITERS_PER_SECOND = 1

class SINDy(FullyObservableModel):
    R"""
    Sparse Identification of Nonlinear Dynamics (SINDy) is an system identification approach that works as follows. 
    Using a library of :math:`k` pre-selected functions (e.g. :math:`f \in \mathbb{R}^k`), it computes numerically the derivatives
    of the system states (e.g. :math:`\dot{x} \in \mathbb{R}^n`) and iteratively solves a least-squares optimization 
    to identify the weights :math:`K \in \mathbb{R}^{n \times k}` that best fit the data: e.g. :math:`\|\dot{x} - Kf(x) \|^2`. 
    In every iteration, functions whose weights are below a user-specified threshold :math:`\lambda` are discarded. 
    For more information, the reader is referred to https://arxiv.org/pdf/2004.08424.pdf

    Parameters:

    - **allow_cross_terms** *(Type: bool, Default: False)* Controls whether to allow polynomial cross-terms and trig
      interaction terms in the basis functions.  For large systems, setting this to true can lead to very large sets
      of basis functions, which can greatly slow down tuning/training.
    
    Hyperparameters:

    - **time_mode** *(Type str, Choices: ["discrete", "continuous"])*: Whether to learn dynamics equations
      as discrete-time or continous-time.
    - **threshold** *(Type: float, Low: 1e-5, High: 10, Default: 1e-2)*: Threshold :math:`\lambda` to use
      for dropping basis functions.
    - **poly_basis** *(Type: bool)*: Whether to use polynomial basis functions
    - **poly_degree** *(Type: int, Low: 2, High: 8, Default: 3)*: Maximum degree of polynomial terms.
      (Conditioned on poly_basis="true")
    - **poly_cross_terms** *(Type: bool)*: Whether to include polynomial cross-terms.
      (Conditioned on poly_basis="true")
    - **trig_basis** *(Type: bool)*: Whether to include trigonometric basis terms.
    - **trig_freq** *(Type: int, Low: 1, High: 8, Default: 1)*: Maximum trig function frequency to include.
      (Conditioned on trig_basis="true")
    - **trig_interaction** *(Type: bool)*: Whether to include cross-multiplication terms between trig functions
      and other state variables.
    """
    def __init__(self, system, allow_cross_terms=False):
        self.allow_cross_terms = allow_cross_terms
        self.system = system
        self.train_time_budget = None
        super().__init__(system, "SINDy")

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        if self.system.discrete_time:
            time_mode = CSH.CategoricalHyperparameter("time_mode", 
                choices=["discrete"])
        else:
            time_mode = CSH.CategoricalHyperparameter("time_mode", 
                choices=["discrete", "continuous"])
        threshold = CSH.UniformFloatHyperparameter("threshold",
                lower=1e-5, upper=1e1, default_value=1e-2, log=True)

        poly_basis = CSH.CategoricalHyperparameter("poly_basis", 
                choices=["true", "false"], default_value="false")
        poly_degree = CSH.UniformIntegerHyperparameter("poly_degree", lower=2, upper=8,
                default_value=3)
        use_poly_degree = CSC.InCondition(child=poly_degree, parent=poly_basis,
                values=["true"])
        if self.allow_cross_terms:
            poly_cross_terms = CSH.CategoricalHyperparameter("poly_cross_terms",
                    choices=["true", "false"], default_value="false")
        else:
            poly_cross_terms = CSH.CategoricalHyperparameter("poly_cross_terms",
                    choices=["false"], default_value="false")

        trig_basis = CSH.CategoricalHyperparameter("trig_basis", 
                choices=["true", "false"], default_value="false")
        trig_freq = CSH.UniformIntegerHyperparameter("trig_freq", lower=1, upper=8,
                default_value=1)
        if self.allow_cross_terms:
            trig_interaction = CSH.CategoricalHyperparameter("trig_interaction", 
                    choices=["true", "false"], default_value="false")
        else:
            trig_interaction = CSH.CategoricalHyperparameter("trig_interaction", 
                    choices=["false"], default_value="false")
        use_trig_freq = CSC.InCondition(child=trig_freq, parent=trig_basis,
                values=["true"])
        use_trig_interaction = CSC.InCondition(child=trig_interaction, parent=trig_basis,
                values=["true"])

        cs.add_hyperparameters([threshold,
            poly_basis, poly_degree, trig_basis, trig_freq, trig_interaction, 
            poly_cross_terms, time_mode])
        cs.add_conditions([use_poly_degree, use_trig_freq, use_trig_interaction])

        return cs

    def set_config(self, config):
        self.poly_basis = get_hyper_bool(config, "poly_basis")
        self.poly_degree = get_hyper_int(config, "poly_degree")
        self.poly_cross_terms = get_hyper_bool(config, "poly_cross_terms")
        self.trig_basis = get_hyper_bool(config, "trig_basis")
        self.trig_freq = get_hyper_int(config, "trig_freq")
        self.trig_interaction = get_hyper_bool(config, "trig_interaction")
        self.time_mode = get_hyper_str(config, "time_mode")
        self.threshold = get_hyper_float(config, "threshold")

    def clear(self):
        self.model = None
        self.is_trained = False
    
    def set_train_budget(self, seconds=None):
        self.train_time_budget = seconds

    def _init_model(self):
        basis_funcs = [IdentityBasisFunction()]
        if self.trig_basis:
            for freq in range(1,self.trig_freq+1):
                basis_funcs += get_trig_basis_funcs(freq)
                if self.trig_interaction:
                    basis_funcs += get_trig_interaction_terms(freq)

        if self.poly_basis:
            for deg in range(2,self.poly_degree+1):
                basis_funcs.append(PolyBasisFunction(deg))
            if self.poly_cross_terms:
                for deg in range(2,self.poly_degree+1):
                    basis_funcs += get_cross_term_basis_funcs(deg)

        library_functions = basis_funcs #[basis.func for basis in basis_funcs]
        function_names = [basis.name_func for basis in basis_funcs]
        library = ps.CustomLibrary(library_functions=library_functions,
                function_names=function_names)
        self.basis_funcs = basis_funcs

        max_iter = 20 if self.train_time_budget is None else int(self.train_time_budget*STLSQ_ITERS_PER_SECOND)
        if self.time_mode == "continuous":
            sindy_model = ps.SINDy(feature_library=library, 
                    discrete_time=False,
                    optimizer=ps.STLSQ(threshold=self.threshold, max_iter=max_iter))
        elif self.time_mode == "discrete":
            sindy_model = ps.SINDy(feature_library=library, 
                    discrete_time=True,
                    optimizer=ps.STLSQ(threshold=self.threshold, max_iter=max_iter))
        self.model = sindy_model

    def train(self, trajs, xdot=None, silent=False):
        self._init_model()

        X = [traj.obs for traj in trajs]
        U = [traj.ctrls for traj in trajs]

        if self.time_mode == "continuous":
            self.model.fit(X, u=U, multiple_trajectories=True, 
                    t=self.system.dt, x_dot=xdot)
        elif self.time_mode == "discrete":
            self.model.fit(X, u=U, multiple_trajectories=True)

        self.is_trained = True

    def pred(self, state, ctrl):
        xpred = self.pred_batch(state.reshape((1,state.size)), 
                ctrl.reshape((1,ctrl.size)))[0,:]
        return xpred

    def pred_batch(self, states, ctrls):
        if self.time_mode == "discrete":
            xpreds = self.model.predict(states, ctrls)
        else:
            pred_dxs = self.model.predict(states, ctrls)
            xpreds = states + self.system.dt * pred_dxs
        return xpreds

    def pred_diff(self, state, ctrl):
        pred, state_jac, ctrl_jac = self.pred_diff_batch(
                state.reshape((1,-1)), ctrl.reshape((1,-1)))
        pred = pred[0]
        state_jac = state_jac[0]
        ctrl_jac = ctrl_jac[0]
        return pred, state_jac, ctrl_jac

    def compute_gradient(self, states, ctrls, basis, coeff, feat_names):
        input_dim = self.state_dim + self.system.ctrl_dim
        idxs = np.mgrid[tuple(slice(input_dim) 
                                for _ in range(basis.n_args))]
        idxs = idxs.reshape((basis.n_args, -1))
        p = states.shape[0]
        state_jac = np.zeros((p,self.state_dim))
        ctrl_jac = np.zeros((p,self.system.ctrl_dim))
        for i in range(idxs.shape[1]):
            # Find feature
            var_names = ["x{}".format(j) if j < self.state_dim else 
                    "u{}".format(j-self.state_dim) for j in idxs[:,i]]
            feat_name = basis.name_func(*var_names)
            try:
                coeff_idx = feat_names.index(feat_name)
            except ValueError:
                continue

            # Compute gradient
            vals = []
            for j in range(idxs.shape[0]):
                idx = idxs[j,i]
                if idx < self.state_dim:
                    val = states[:,idx]
                else:
                    val = ctrls[:,idx-self.state_dim]
                vals.append(val)
            grads = basis.grad_func(*vals)
            grads = np.array(grads)
            for j in range(idxs.shape[0]):
                idx = idxs[j,i]
                if idx < self.state_dim:
                    state_jac[:,idx] += coeff[coeff_idx] * grads[j]
                else:
                    ctrl_jac[:,idx-self.state_dim] += coeff[coeff_idx] * grads[j]
        return state_jac, ctrl_jac

    def pred_diff_batch(self, states, ctrls):
        xpred = self.pred_batch(states, ctrls)
        p = states.shape[0]
        state_jac = np.zeros((p, self.state_dim, self.state_dim))
        ctrl_jac = np.zeros((p, self.state_dim, self.system.ctrl_dim))
        coeffs = self.model.coefficients()
        feat_names = self.model.get_feature_names()
        for i in range(self.state_dim):
            for basis in self.basis_funcs:
                sj, cj = self.compute_gradient(
                        states, ctrls, basis, coeffs[i,:], feat_names)
                state_jac[:,i,:] += sj
                ctrl_jac[:,i,:] += cj
        if self.time_mode == "continuous":
            ident = np.array([np.eye(self.state_dim) for _ in
                range(p)])
            state_jac = ident + self.system.dt * state_jac
            ctrl_jac = self.system.dt * ctrl_jac
        return xpred, state_jac, ctrl_jac

    def get_parameters(self):
        return {"pickled_model" : pickle.dumps(self.model)}

    def set_parameters(self, params):
        self.model = pickle.loads(params["pickled_model"])
