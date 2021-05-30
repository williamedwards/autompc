import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from pdb import set_trace
from sklearn.linear_model import  Lasso

from .model import Model, ModelFactory
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

import pysindy as ps
import pysindy.differentiation as psd

from .basis_funcs import *

class FourthOrderFiniteDifference(psd.base.BaseDifferentiation):
    def _differentiate(self, x, t):
        fd = psd.FiniteDifference(order=2)
        xdot = fd._differentiate(x, t)
        xdot[2:-2] = -x[4:] + 8 * x[3:-1] - 8 * x[1:-3] + x[:-4]
        return xdot

class SINDyFactory(ModelFactory):
    """
    Sparse Identification of Nonlinear Dynamics (SINDy) is an system identification approach that works as follows. 
    Using a library of $k$ pre-selected functions (e.g. $f \in \mathbb{R}^k$), it computes numerically the derivatives
    of the system states (e.g. $\dot{x} \in \mathbb{R}^n$) iteratively solves a least-squares optimization 
    to identify the weights $K \in \mathbb{R}^{n \times k}$ that best fit the data: e.g. $\|\dot{x} - Kf(x) \|^2$. 
    In every iteration, functions whose weights are below a user-specified threshold $\lambda$ are discarded. 
    For more information, the reader is referred to https://arxiv.org/pdf/2004.08424.pdf
    
    
    SINDy Docs :math:`a^2 + b^2 = c^2`.

    .. math::
      
       \\frac{4}{3} \pi r^3 = V

       x + y = z

    Hyperparameters:

    - *time_mode* (Type str, Choices: ["discrete", "continuous"]): Whether to learn dynamics equations
      as discrete-time or continous-time.
    - *method* (Type str, Choices: ["lstsq, lasso"], Default: "lstsq"): Method for selecting
      model coefficients.
    - *lasso_alpha* (Type: str, Low: 10^-5, High: 10^2, Default: 1): α parameter for lasso regression.
      (Conditioned on method="lasso")
    - *poly_basis* (Type: bool): Whether to use polynomial basis functions
    - *poly_degree* (Type: int, Low: 2, High: 8, Default: 3): Maximum degree of polynomial terms.
      (Conditioned on poly_basis="true")
    - *poly_cross_terms* (Type: bool): Whether to include polynomial cross-terms.
      (Conditioned on poly_basis="true")
    - *trig_basis* (Type: bool): Whether to include trigonometric basis terms.
    - *trig_freq* (Type: int, Low: 1, High: 8, Default: 1): Maximum trig function frequency to include.
      (Conditioned on trig_basis="true")
    - *trig_interaction* (Type: bool): Whether to include cross-multiplication terms between trig functions
      and other state variables.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Model = SINDy
        self.name = "SINDy"

    def get_configuration_space(system):
        cs = CS.ConfigurationSpace()
        time_mode = CSH.CategoricalHyperparameter("time_mode", 
                choices=["discrete", "continuous"])
        method = CSH.CategoricalHyperparameter("method", choices=["lstsq", "lasso"])
        threshold = CSH.UniformFloatHyperparameter("threshold",
                lower=1e-5, upper=1e1, default_value=1e-2, log=True)
        lasso_alpha = CSH.UniformFloatHyperparameter("lasso_alpha", 
                lower=1e-5, upper=1e2, default_value=1.0, log=True)
        use_lasso_alpha = CSC.InCondition(child=lasso_alpha, parent=method, 
                values=["lasso"])

        poly_basis = CSH.CategoricalHyperparameter("poly_basis", 
                choices=["true", "false"], default_value="false")
        poly_degree = CSH.UniformIntegerHyperparameter("poly_degree", lower=2, upper=8,
                default_value=3)
        use_poly_degree = CSC.InCondition(child=poly_degree, parent=poly_basis,
                values=["true"])
        poly_cross_terms = CSH.CategoricalHyperparameter("poly_cross_terms",
                choices=["true", "false"], default_value="false")

        trig_basis = CSH.CategoricalHyperparameter("trig_basis", 
                choices=["true", "false"], default_value="false")
        trig_freq = CSH.UniformIntegerHyperparameter("trig_freq", lower=1, upper=8,
                default_value=1)
        trig_interaction = CSH.CategoricalHyperparameter("trig_interaction", 
                choices=["true", "false"], default_value="false")
        use_trig_freq = CSC.InCondition(child=trig_freq, parent=trig_basis,
                values=["true"])
        use_trig_interaction = CSC.InCondition(child=trig_interaction, parent=trig_basis,
                values=["true"])

        cs.add_hyperparameters([method, lasso_alpha, threshold,
            poly_basis, poly_degree, trig_basis, trig_freq, trig_interaction, 
            poly_cross_terms, time_mode])
        cs.add_conditions([use_lasso_alpha, use_poly_degree, use_trig_freq, use_trig_interaction])

        return cs

class SINDy(Model):
    def __init__(self, system, method, lasso_alpha=None, threshold=1e-2, poly_basis=False,
            poly_degree=1, poly_cross_terms=False, trig_basis=False, trig_freq=1, trig_interaction=False, time_mode="discrete"):
        super().__init__(system)

        self.method = method
        self.lasso_alpha = lasso_alpha
        if type(poly_basis) == str:
            poly_basis = True if poly_basis == "true" else False
        self.poly_basis = poly_basis
        self.poly_degree = poly_degree
        self.poly_cross_terms = poly_cross_terms
        if type(trig_basis) == str:
            trig_basis = True if trig_basis == "true" else False
        self.trig_basis = trig_basis
        self.trig_freq = trig_freq
        self.trig_interaction = trig_interaction
        if type(trig_interaction) == str:
            self.trig_interaction = True if trig_interaction == "true" else False
        self.time_mode = time_mode
        self.threshold = threshold

    @staticmethod
    def get_configuration_space(system):


        return cs

    def traj_to_state(self, traj):
        return traj[-1].obs.copy()
    
    def update_state(self, state, new_ctrl, new_obs):
        return new_obs.copy()

    @property
    def state_dim(self):
        return self.system.obs_dim

    def train(self, trajs, xdot=None, silent=False):
        X = [traj.obs for traj in trajs]
        U = [traj.ctrls for traj in trajs]

        #basis_funcs = [get_constant_basis_func(), get_identity_basis_func()]
        basis_funcs = [get_identity_basis_func()]
        if self.trig_basis:
            for freq in range(1,self.trig_freq+1):
                basis_funcs += get_trig_basis_funcs(freq)
                if self.trig_interaction:
                    basis_funcs += get_trig_interaction_terms(freq)

        if self.poly_basis:
            for deg in range(2,self.poly_degree+1):
                basis_funcs.append(get_poly_basis_func(deg))
            if self.poly_cross_terms:
                for deg in range(2,self.poly_degree+1):
                    basis_funcs += get_cross_term_basis_funcs(deg)

        library_functions = [basis.func for basis in basis_funcs]
        function_names = [basis.name_func for basis in basis_funcs]
        library = ps.CustomLibrary(library_functions=library_functions,
                function_names=function_names)
        self.basis_funcs = basis_funcs

        if self.time_mode == "continuous":
            sindy_model = ps.SINDy(feature_library=library, 
                    discrete_time=False,
                    optimizer=ps.STLSQ(threshold=self.threshold))
            sindy_model.fit(X, u=U, multiple_trajectories=True, 
                    t=self.system.dt, x_dot=xdot)
        elif self.time_mode == "discrete":
            sindy_model = ps.SINDy(feature_library=library, 
                    discrete_time=True,
                    optimizer=ps.STLSQ(threshold=self.threshold))
            sindy_model.fit(X, u=U, multiple_trajectories=True)
        self.model = sindy_model

    def pred(self, state, ctrl):
        xpred = self.pred_parallel(state.reshape((1,state.size)), 
                ctrl.reshape((1,ctrl.size)))[0,:]
        return xpred

    def pred_parallel(self, states, ctrls):
        if self.time_mode == "discrete":
            xpreds = self.model.predict(states, ctrls)
        else:
            pred_dxs = self.model.predict(states, ctrls)
            xpreds = states + self.system.dt * pred_dxs
        return xpreds

    def pred_diff(self, state, ctrl):
        pred, state_jac, ctrl_jac = self.pred_diff_parallel(
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

    def pred_diff_parallel(self, states, ctrls):
        xpred = self.pred_parallel(states, ctrls)
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

    # TODO fix this
    def get_parameters(self):
        return {"A" : np.copy(self.A),
                "B" : np.copy(self.B)}

    def set_parameters(self, params):
        self.A = np.copy(params["A"])
        self.B = np.copy(params["B"])
