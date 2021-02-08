import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from pdb import set_trace
from sklearn.linear_model import  Lasso

from ..model import Model
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

import pysindy as ps

from .basis_funcs import *

class SINDy(Model):
    def __init__(self, system, method, lasso_alpha_log10=None, poly_basis=False,
            poly_degree=1, poly_cross_terms=False, trig_basis=False, trig_freq=1, time_mode="discrete"):
        super().__init__(system)

        self.method = method
        if not lasso_alpha_log10 is None:
            self.lasso_alpha = 10**lasso_alpha_log10
        else:
            self.lasso_alpha = None
        if type(poly_basis) == str:
            poly_basis = True if poly_basis == "true" else False
        self.poly_basis = poly_basis
        self.poly_degree = poly_degree
        self.poly_cross_terms = poly_cross_terms
        if type(trig_basis) == str:
            trig_basis = True if trig_basis == "true" else False
        self.trig_basis = trig_basis
        self.trig_freq = trig_freq
        self.trig_interaction = False
        self.time_mode = time_mode

    @staticmethod
    def get_configuration_space(system):
        cs = CS.ConfigurationSpace()
        time_mode = CSH.CategoricalHyperparameter("time_mode", 
                choices=["discrete", "continuous"])
        method = CSH.CategoricalHyperparameter("method", choices=["lstsq", "lasso"])
        lasso_alpha_log10 = CSH.UniformFloatHyperparameter("lasso_alpha_log10", 
                lower=-5.0, upper=2.0, default_value=0.0)
        use_lasso_alpha = CSC.InCondition(child=lasso_alpha_log10, parent=method, 
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
        use_trig_freq = CSC.InCondition(child=trig_freq, parent=trig_basis,
                values=["true"])

        cs.add_hyperparameters([method, lasso_alpha_log10, 
            poly_basis, poly_degree, trig_basis, trig_freq, 
            poly_cross_terms, time_mode])
        cs.add_conditions([use_lasso_alpha, use_poly_degree, use_trig_freq])

        return cs

    def traj_to_state(self, traj):
        return traj[-1].obs.copy()
    
    def update_state(self, state, new_ctrl, new_obs):
        return new_obs.copy()

    @property
    def state_dim(self):
        return self.system.obs_dim

    def train(self, trajs, xdot=None):
        X = [traj.obs for traj in trajs]
        U = [traj.ctrls for traj in trajs]

        basis_funcs = [get_identity_basis_func()]
        if self.trig_basis:
            for freq in range(1,self.trig_freq+1):
                basis_funcs += get_trig_basis_funcs(freq)

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
                    optimizer=ps.STLSQ(threshold=0.01))
            sindy_model.fit(X, u=U, multiple_trajectories=True, 
                    t=self.system.dt, x_dot=xdot)
        elif self.time_mode == "discrete":
            sindy_model = ps.SINDy(feature_library=library, 
                    discrete_time=True,
                    optimizer=ps.STLSQ(threshold=0.01))
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
            print("basis.n_args=", basis.n_args)
        return state_jac, ctrl_jac

    def pred_diff_parallel(self, states, ctrls):
        xpred = self.pred(states, ctrls)
        p = states.shape[0]
        state_jac = np.zeros((p, self.state_dim, self.state_dim))
        ctrl_jac = np.zeros((p, self.state_dim, self.system.ctrl_dim))
        coeffs = self.model.coefficients()
        feat_names = self.model.get_feature_names()
        for i in range(self.state_dim):
            for basis in self.basis_funcs:
                print("A:", basis.n_args)
                sj, cj = self.compute_gradient(
                        states, ctrls, basis, coeffs[i,:], feat_names)
                state_jac[:,i,:] += sj
                ctrl_jac[:,i,:] += cj
        if self.time_mode == "continuous":
            ident = np.array([np.eye(self.state_dim) for _ in
                range(p)])
            state_jac = ident + self.system.dt * state_jac
            ctrl_jac = self.system.dt * ctrl_jac
        set_trace()
        return xpred, state_jac, ctrl_jac

    # TODO fix this
    def get_parameters(self):
        return {"A" : np.copy(self.A),
                "B" : np.copy(self.B)}

    def set_parameters(self, params):
        self.A = np.copy(params["A"])
        self.B = np.copy(params["B"])
