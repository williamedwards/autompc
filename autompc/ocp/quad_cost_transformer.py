# Created by William Edwards (wre2@illinois.edu), 2021-01-24

# Standard library includes
import copy
from collections import defaultdict

# External library includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

# Internal library includes
from .ocp_transformer import OCPTransformer,PrototypeOCP
from ..costs.quad_cost import QuadCost

QUAD_COST_DEFAULT_BOUNDS = (1e-3, 1e4)
QUAD_COST_DEFAULT_VALUE = 1.0
QUAD_COST_DEFAULT_LOG = True
class QuadCostTransformer(OCPTransformer):
    """
    Transformer to throw out original OCP's cost and produce quadratic cost. 
    See :class:`QuadCost`.

    Parameters:
     - *goal* (numpy array of size system.obs_dim): Goal state. Default is
        0 state.

    Hyperparameters:

     - * **x**_Q* (float, Lower: 10^-3, Upper: 10^4): Digaonal Q matrix value
        corresponding to observation dimension with label **x**
     - * **x**_R* (float, Lower: 10^-3, Upper: 10^4): Digaonal R matrix value
        corresponding to control dimension with label **x**
     - * **x**_F* (float, Lower: 10^-3, Upper: 10^4): Digaonal F matrix value
        corresponding to ovservation dimension with label **x**
    """
    def __init__(self, system, goal=None):
        self._goal_tunable = dict() # Key: obsname, Value: (lower, upper, default, log_scale)
        self._goal = goal
        super().__init__(system, "QuadCostTransformer")

    def set_tunable_goal(self, obsname, lower_bound, upper_bound, default, log=False):
        """
        Allow part of the goal to be chosen by hyperparameters.

        Parameters:
        - *obsname* (str): Name of observation dimension 
        - *lower_bound* (float): Lower bound for goal hyperparameter
        - *upper_bound* (float): Upper bound for goal hyperparameter
        - *default* (float): Default value for the goal hyperparameter
        - *log* (bool): Whether hyperparameter should use logarithmic scale.  (Default: False)
        """
        if not obsname in self.system.observations:
            raise ValueError("obsname not recognized")
        hyper = CSH.UniformFloatHyperparameter(obsname+"_Goal",
                lower=lower_bound, upper=upper_bound, default_value=default, log=log)
        cs = self.get_config_space()
        cs.add_hyperparameter(hyper)
        self.set_configuration_space(cs)

    def fix_Q_value(self, obsname, value):
        """
        Fix an entry in the Q matrix to a constant value.

        Parameters:
        - *obsname* (str): Name of observation dimension 
        - *value* (float): Fixed value for matrix entry
        """
        args = {obsname+'_Q':value}
        self.fix_hyperparameters(**args)

    def fix_R_value(self, ctrlname, value):
        """
        Fix an entry in the R matrix to a constant value.

        Parameters:
        - *ctrlname* (str): Name of observation dimension 
        - *value* (float): Fixed value for matrix entry
        """
        args = {ctrlname+'_R':value}
        self.fix_hyperparameters(**args)

    def fix_F_value(self, obsname, value):
        """
        Fix an entry in the F matrix to a constant value.

        Parameters:
        - *obsname* (str): Name of observation dimension 
        - *value* (float): Fixed value for matrix entry
        """
        args = {obsname+'_F':value}
        self.fix_hyperparameters(**args)

    def set_Q_bounds(self, obsname, lower_bound, upper_bound, default, log=False):
        """
        Set the upper and lower bounds for a hyperparameter controlling
        an entry of the Q matrix.
        Parameters:
        - *obsname* (str): Name of observation dimension. If None, change is applied
                           to all observation dimensions.
        - *lower_bound* (float): Lower bound for the hyperparameter
        - *upper_bound* (float): Upper bound for the hyperparameter
        - *default* (float): Default value for the hyperparameter.
        - *log* (bool): Whether hyperparameter should use logarithmic scale.  (Default: False)
        """
        args = {obsname+'_Q':(lower_bound,upper_bound)}
        self.set_hyperparameter_bounds(**args)
        args[obsname+'_Q'] = default
        self.set_hyperparameter_defaults(**args)
        args[obsname+'_Q'] = log
        self.set_hyperparameter_logs(**args)

    def set_R_bounds(self, ctrlname, lower_bound, upper_bound, default, log=False):
        """
        Set the upper and lower bounds for a hyperparameter controlling
        an entry of the R matrix.
        Parameters:
        - *ctrlname* (str): Name of control dimension. If None, change is applied
                           to all control dimensions.
        - *lower_bound* (float): Lower bound for the hyperparameter
        - *upper_bound* (float): Upper bound for the hyperparameter
        - *default* (float): Default value for the hyperparameter.
        - *log* (bool): Whether hyperparameter should use logarithmic scale.  (Default: False)
        """
        args = {ctrlname+'_R':(lower_bound,upper_bound)}
        self.set_hyperparameter_bounds(**args)
        args[ctrlname+'_R'] = default
        self.set_hyperparameter_defaults(**args)
        args[ctrlname+'_R'] = log
        self.set_hyperparameter_logs(**args)

    def set_F_bounds(self, obsname, lower_bound, upper_bound, default, log=False):
        """
        Set the upper and lower bounds for a hyperparameter controlling
        an entry of the F matrix.
        Parameters:
        - *obsname* (str): Name of observation dimension. If None, change is applied
                           to all observation dimensions.
        - *lower_bound* (float): Lower bound for the hyperparameter
        - *upper_bound* (float): Upper bound for the hyperparameter
        - *default* (float): Default value for the hyperparameter.
        - *log* (bool): Whether hyperparameter should use logarithmic scale.  (Default: False)
        """
        args = {obsname+'_F':(lower_bound,upper_bound)}
        self.set_hyperparameter_bounds(**args)
        args[obsname+'_F'] = default
        self.set_hyperparameter_defaults(**args)
        args[obsname+'_F'] = log
        self.set_hyperparameter_logs(**args)

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        for name in self.system.observations:
            hyper = CSH.UniformFloatHyperparameter(name+"_Q",
                    lower=QUAD_COST_DEFAULT_BOUNDS[0], upper=QUAD_COST_DEFAULT_BOUNDS[1], default_value=QUAD_COST_DEFAULT_VALUE, log=QUAD_COST_DEFAULT_LOG)
            cs.add_hyperparameter(hyper)
        for name in self.system.controls:
            hyper = CSH.UniformFloatHyperparameter(name+"_R",
                    lower=QUAD_COST_DEFAULT_BOUNDS[0], upper=QUAD_COST_DEFAULT_BOUNDS[1], default_value=QUAD_COST_DEFAULT_VALUE, log=QUAD_COST_DEFAULT_LOG)
            cs.add_hyperparameter(hyper)
        for name in self.system.observations:
            hyper = CSH.UniformFloatHyperparameter(name+"_F",
                    lower=QUAD_COST_DEFAULT_BOUNDS[0], upper=QUAD_COST_DEFAULT_BOUNDS[1], default_value=QUAD_COST_DEFAULT_VALUE, log=QUAD_COST_DEFAULT_LOG)
            cs.add_hyperparameter(hyper)
        for obs_name in self._goal_tunable.keys():
            (lower,upper,defaults,log) = self._goal_tunable[name]
            hyper = CSH.UniformFloatHyperparameter(name+"_Goal",
                    lower=lower, upper=upper, default_value=defaults, log=log)
            cs.add_hyperparameter(hyper)
        return cs

    def is_compatible(self, ocp):
        return True

    def _get_matrix(self, cfg, label, fixed_dict, dim_names):
        mat = np.zeros((len(dim_names), len(dim_names)))
        for i, name in enumerate(dim_names):
            hyper_name = f"{name}_{label}"
            assert hyper_name in cfg
            mat[i,i] = cfg[hyper_name]
        return mat

    def get_prototype(self, config, ocp):
        return PrototypeOCP(ocp, cost=QuadCost)

    def _get_cost_matrices(self):
        config = self.get_config()

        Qdiag = [config[name+"_Q"] for name in self.system.observations]
        Rdiag = [config[name+"_R"] for name in self.system.controls]
        Fdiag = [config[name+"_F"] for name in self.system.observations]

        Q = np.diag(Qdiag)
        R = np.diag(Rdiag)
        F = np.diag(Fdiag)

        return Q, R, F

    def _get_goal(self, ocp):
        config = self.get_config()

        if self._goal is not None:
            goal = np.copy(self._goal)
        else:
            goal = np.copy(ocp.cost.goal)

        for index, obs_name in enumerate(self.system.observations):
            if f"{obs_name}_Goal" in config:
                goal[index] = config[f"{obs_name}_Goal"]

        return goal
        

    def __call__(self, ocp):
        Q, R, F = self._get_cost_matrices()
        goal = self._get_goal(ocp)

        new_cost = QuadCost(self.system, Q, R, F, goal=goal)
        new_ocp = copy.deepcopy(ocp)
        new_ocp.set_cost(new_cost)

        return new_ocp
