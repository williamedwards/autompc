# Standard library includes
from itertools import combinations

# External library includes
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from sklearn.linear_model import  Lasso
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

# Internal library includes
from ..system import System
from .model import Model
from .stable_koopman import stabilize_discrete
from .basis_funcs import *
from ..utils.cs_utils import *

#estimate of how many iters of Lasso should be performed, max_iters = budget * this constant / N
LASSO_ITERS_PER_SECOND_BY_DATAPOINT = 20.0*100000.0

class Koopman(Model):
    """
    This class identifies Koopman models of the form :math:`\dot{\Psi}(x) = A\Psi(x) + Bu`. 
    Given states :math:`x \in \mathbb{R}^n`, :math:`\Psi(x) \in \mathbb{R}^N` ---termed Koopman 
    basis functions--- are used to lift the dynamics in a higher-dimensional space
    where nonlinear functions of the system states evolve linearly. The identification
    of the Koopman model, specified by the A and B matrices, can be done in various ways, 
    such as solving a least squares solution :math:`\|\dot{\Psi}(x) - [A, B][\Psi(x),u]^T\|`.
    
    The choice of the basis functions is an open research question. In this implementation, 
    we choose basis functions that depend only on the system states and not the control, 
    in order to derive a representation that is amenable for LQR control. 

    Parameters:

    - **allow_cross_terms** *(Type: bool, Default: False)* Controls whether to allow polynomial cross-terms and trig
      interaction terms in the basis functions.  For large systems, setting this to true can lead to very large sets
      of basis functions, which can greatly slow down tuning/training.

    Hyperparameters:

    - **method** *(Type: str, Choices: ["lstsq", "lasso", "stable"])*: Method for training Koopman
      operator.
    - **lasso_alpha** *(Type: float, Low: 10^-10, High: 10^2, Defalt: 1.0)*: α parameter for Lasso
      regression. (Conditioned on method="lasso").
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
        self.budget = None
        super().__init__(system, "Koopman")

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        method = CSH.CategoricalHyperparameter("method", choices=["lstsq", "lasso",
            "stable"])
        lasso_alpha = CSH.UniformFloatHyperparameter("lasso_alpha", 
                lower=1e-10, upper=1e2, default_value=1.0, log=True)
        use_lasso_alpha = CSC.InCondition(child=lasso_alpha, parent=method, 
                values=["lasso"])

        poly_basis = CSH.CategoricalHyperparameter("poly_basis", 
                choices=["true", "false"], default_value="false")
        poly_degree = CSH.UniformIntegerHyperparameter("poly_degree", lower=2, upper=8,
                default_value=3)
        if self.allow_cross_terms:
            poly_cross_terms = CSH.CategoricalHyperparameter("poly_cross_terms",
                    choices=["true", "false"], default_value="false")
        else:
            poly_cross_terms = CSH.CategoricalHyperparameter("poly_cross_terms",
                    choices=["false"], default_value="false")
        use_poly_degree = CSC.InCondition(child=poly_degree, parent=poly_basis,
                values=["true"])
        use_poly_cross_terms = CSC.InCondition(child=poly_cross_terms, parent=poly_basis,
                values=["true"])

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

        cs.add_hyperparameters([method, poly_basis, poly_degree,
            trig_basis, trig_freq, poly_cross_terms, trig_interaction, lasso_alpha])
        cs.add_conditions([use_poly_degree, use_trig_freq, use_lasso_alpha,
            use_poly_cross_terms, use_trig_interaction])

        return cs

    def set_config(self, config):
        self.method = get_hyper_str(config, "method")
        self.lasso_alpha = get_hyper_float(config, "lasso_alpha")
        self.poly_basis = get_hyper_bool(config, "poly_basis")
        self.poly_degree = get_hyper_int(config, "poly_degree")
        self.poly_cross_terms = get_hyper_bool(config, "poly_cross_terms")
        self.trig_basis = get_hyper_bool(config, "trig_basis")
        self.trig_freq = get_hyper_int(config, "trig_freq")
        self.trig_interaction = get_hyper_bool(config, "trig_interaction")

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
        self.basis_funcs = basis_funcs

    def _apply_basis(self, state):
        tr_state = []
        for basis in self.basis_funcs:
            for idxs in combinations(range(self.system.obs_dim), basis.n_args):
                tr_state.append(basis(*state[list(idxs)])) 
        return np.array(tr_state)

    def _transform_observations(self, observations):
        return np.apply_along_axis(self._apply_basis, 1, observations)

    def traj_to_state(self, traj):
        return self._transform_observations(traj.obs[:])[-1,:]

    def traj_to_states(self, traj):
        return self._transform_observations(traj.obs[:])
    
    def update_state(self, state, new_ctrl, new_obs):
        return self._apply_basis(new_obs)

    @property
    def state_dim(self):
        state_dim = 0
        for basis in self.basis_funcs:
            for idxs in combinations(range(self.system.obs_dim), basis.n_args):
                state_dim += 1
        return state_dim
    
    @property
    def state_system(self):
        vars = []
        for basis in self.basis_funcs:
            for idxs in combinations(range(self.system.obs_dim), basis.n_args):
                vars.append(basis.name_func(*[self.system.observations[i] for i in idxs]))
        return System(vars,self.system.controls)

    def set_train_budget(self, seconds=None):
        self.budget = seconds

    def train(self, trajs, silent=False):
        trans_obs = [self._transform_observations(traj.obs[:]) for traj in trajs]
        X = np.concatenate([obs[:-1,:] for obs in trans_obs]).T
        Y = np.concatenate([obs[1:,:] for obs in trans_obs]).T
        U = np.concatenate([traj.ctrls[:-1,:] for traj in trajs]).T
        
        n = X.shape[0] # state dimension
        m = U.shape[0] # control dimension    
        
        XU = np.concatenate((X, U), axis = 0) # stack X and U together
        if self.method == "lstsq": # Least Squares Solution
            AB = np.dot(Y, sla.pinv2(XU))
            A = AB[:n, :n]
            B = AB[:n, n:]
        elif self.method == "lasso":  # Call lasso regression on coefficients
            max_iter = 1000 if self.budget is None else max(20,int(self.budget*LASSO_ITERS_PER_SECOND_BY_DATAPOINT/n) )
            print("Call Lasso (%d iters)"%max_iter)
            clf = Lasso(alpha=self.lasso_alpha, max_iter=max_iter)
            clf.fit(XU.T, Y.T)
            AB = clf.coef_
            A = AB[:n, :n]
            B = AB[:n, n:]
        elif self.method == "stable": # Compute stable A, and B
            print("Compute Stable Koopman")
            # call function
            A, _, _, _, B, _ = stabilize_discrete(X, U, Y, time_budget=self.budget)
            A = np.real(A)
            B = np.real(B)

        self.A, self.B = A, B
        self.is_trained = True

    def clear(self):
        self.A = None
        self.B = None
        self.is_trained = False

    def pred(self, state, ctrl):
        xpred = self.A @ state + self.B @ ctrl
        return xpred

    def pred_batch(self, states, ctrls):
        statesnew = self.A @ states.T + self.B @ ctrls.T

        return statesnew.T

    def pred_diff(self, state, ctrl):
        xpred = self.A @ state + self.B @ ctrl

        return xpred, np.copy(self.A), np.copy(self.B)

    def to_linear(self):
        return np.copy(self.A), np.copy(self.B), None

    def get_parameters(self):
        return {"A" : self.A.tolist(),
                "B" : self.B.tolist()}

    def set_parameters(self, params):
        self.A = np.array(params["A"])
        self.B = np.array(params["B"])
