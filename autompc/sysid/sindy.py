from functools import partial
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

class SINDy(Model):
    def __init__(self, system, method, lasso_alpha_log10=None, poly_basis=False,
            poly_degree=1, trig_basis=False, trig_freq=1):
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
        if type(trig_basis) == str:
            trig_basis = True if trig_basis == "true" else False
        self.trig_basis = trig_basis
        self.trig_freq = trig_freq

    @staticmethod
    def get_configuration_space(system):
        cs = CS.ConfigurationSpace()
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

        trig_basis = CSH.CategoricalHyperparameter("trig_basis", 
                choices=["true", "false"], default_value="false")
        trig_freq = CSH.UniformIntegerHyperparameter("trig_freq", lower=1, upper=8,
                default_value=1)
        use_trig_freq = CSC.InCondition(child=trig_freq, parent=trig_basis,
                values=["true"])

        cs.add_hyperparameters([method, lasso_alpha_log10, poly_basis, poly_degree,
            trig_basis, trig_freq])
        cs.add_conditions([use_lasso_alpha, use_poly_degree, use_trig_freq])

        return cs

    def traj_to_state(self, traj):
        return traj[-1].obs.copy()
    
    def update_state(self, state, new_ctrl, new_obs):
        return new_obs.copy()

    @property
    def state_dim(self):
        return self.system.obs_dim

    def train(self, trajs):
        X = [traj.obs for traj in trajs]
        U = [traj.ctrls for traj in trajs]

        library_functions = [
                lambda x : x
        ]
        function_gradients = [
                lambda x : 1
        ]
        function_names = [
                lambda x : x
        ]

        if self.trig_basis:
            for freq in range(1,self.trig_freq+1):
                library_functions += [
                    (lambda f: lambda x : np.sin(f * x))(freq),
                    (lambda f: lambda x : np.cos(f * x))(freq)
                ]
                function_gradients += [
                    (lambda f: lambda x : f * np.cos(f * x))(freq),
                    (lambda f: lambda x : f * -np.sin(f * x))(freq)
                ]
                function_names += [
                    (lambda f: lambda x : "sin({} {})".format(f, x))(freq),
                    (lambda f: lambda x : "cos({} {})".format(f, x))(freq),
                ]
        if self.poly_basis:
            for deg in range(2,self.poly_degree+1):
                library_functions += [
                    (lambda d: lambda x : x ** d)(deg)
                ]
                function_gradients += [
                    (lambda d: lambda x : d * x ** (d-1))(deg)
                ]
                function_names += [
                    (lambda d: lambda x, d=d : "{}^{}".format(x, d))(deg)
                ]

        library = ps.CustomLibrary(library_functions=library_functions,
                function_names=function_names)
        self.function_gradients = function_gradients

        sindy_model = ps.SINDy(feature_library=library, discrete_time=True,
                optimizer=ps.STLSQ(threshold=0.000))
        sindy_model.fit(X, u=U, multiple_trajectories=True)
        self.model = sindy_model


    def pred(self, state, ctrl):
        xpred = self.model.predict(state.reshape((1,state.size)), 
                ctrl.reshape((1,ctrl.size)))[0,:]
        return xpred

    def pred_diff(self, state, ctrl):
        xpred = self.model.predict(state.reshape((1,state.size)), 
                ctrl.reshape((1,ctrl.size)))[0,:]
        state_jac = np.zeros((self.state_dim, self.state_dim))
        ctrl_jac = np.zeros((self.state_dim, self.system.ctrl_dim))
        coeff = self.model.coefficients()
        for i in range(self.state_dim):
            coeff_idx = 0
            for gr in self.function_gradients:
                for j in range(self.state_dim):
                    state_jac[i, j] += coeff[i,coeff_idx] * gr(state[j])
                    coeff_idx += 1
                for j in range(self.system.ctrl_dim):
                    ctrl_jac[i, j] += coeff[i,coeff_idx] * gr(ctrl[j])
                    coeff_idx += 1

        return xpred, state_jac, ctrl_jac


    def get_parameters(self):
        return {"A" : np.copy(self.A),
                "B" : np.copy(self.B)}

    def set_parameters(self, params):
        self.A = np.copy(params["A"])
        self.B = np.copy(params["B"])
