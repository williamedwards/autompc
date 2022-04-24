# Created by William Edwards (wre2@illinois.edu), 2021-01-24

# Standard library includes
import copy
from collections import defaultdict

# Internal library includes
from .ocp_factory import OCPFactory
from .ocp import PrototypeOCP
from ..costs.quad_limit_cost import QuadLimitCost

# External library includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

def construct_default_bounds():
    return (1e-3, 1e4, 1.0, True)

class QuadLimitCostFactory(OCPFactory):
    """
    Factory to produce quadratic cost.  This cost has the form

    .. math::

        x_N^T (F - x_g) x_N  + \\sum_{t=0}^{N} (x_t^T (Q - x_g) x_t + u_t^T R u_t)

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
        if goal is None:
            self.goal = None
        else:
            self.goal = goal[:]

        self._Q_bounds = defaultdict(construct_default_bounds) # Key: obsname, Value: (lower, upper, default, log_scale)
        self._R_bounds = defaultdict(construct_default_bounds)
        self._F_bounds = defaultdict(construct_default_bounds)
        self._Q_fixed = dict() # Key: obsname, Value: fixed_value
        self._R_fixed = dict() # Key: obsname, Value: fixed_value
        self._F_fixed = dict() # Key: obsname, Value: fixed_value
        self._goal_tunable = dict() # Key: obsname, Value: (lower, upper, default, log_scale)

        self._scale_bounds = defaultdict(construct_default_bounds) # Key: obsname, Value: (lower, upper, default, log_scale)
        self._limits = dict() # Key: obs/ctrlname, Value: (limit, upper)
        self._scale_fixed = dict() # Key: obs/ctrlname, Value: limit
        super().__init__(system, "QuadLimitCostFactory")

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
        self._goal_tunable[obsname] = (lower_bound, upper_bound, default, log)

    def _fix_value(self, value_dict, allowed_names, dim_name, value):
        if not dim_name is None:
            if not dim_name in allowed_names:
                raise ValueError(f"{dim_name} is not in the system")
            value_dict[dim_name] = value
        else:
            for name in allowed_names:
                value_dict[name] = value

    def _set_bounds(self, value_dict, allowed_names, dim_name, lower_bound, upper_bound, default, log):
        if not dim_name is None:
            if not dim_name in allowed_names:
                raise ValueError(f"{dim_name} is not in the system")
            value_dict[dim_name] = (lower_bound, upper_bound, default, log)
        else:
            for name in allowed_names:
                value_dict[name] = (lower_bound, upper_bound, default, log)

    def fix_Q_value(self, obsname, value):
        """
        Fix an entry in the Q matrix to a constant value.

        Parameters:
        - *obsname* (str): Name of observation dimension 
        - *value* (float): Fixed value for matrix entry
        """
        self._fix_value(self._Q_fixed, self.system.observations, obsname, value)

    def fix_R_value(self, ctrlname, value):
        """
        Fix an entry in the R matrix to a constant value.

        Parameters:
        - *ctrlname* (str): Name of observation dimension 
        - *value* (float): Fixed value for matrix entry
        """
        self._fix_value(self._R_fixed, self.system.controls, ctrlname, value)

    def fix_F_value(self, obsname, value):
        """
        Fix an entry in the F matrix to a constant value.

        Parameters:
        - *obsname* (str): Name of observation dimension 
        - *value* (float): Fixed value for matrix entry
        """
        self._fix_value(self._F_fixed, self.system.observations, obsname, value)

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
        self._set_bounds(self._Q_bounds, self.system.observations, obsname,
            lower_bound, upper_bound, default, log)

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
        self._set_bounds(self._R_bounds, self.system.controls, ctrlname,
            lower_bound, upper_bound, default, log)

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
        self._set_bounds(self._F_bounds, self.system.observations, obsname,
            lower_bound, upper_bound, default, log)

    def _get_hyperparameters(self, label, bounds_dict, fixed_dict, dim_names):
        hyperparameters = []
        for name in dim_names:
            if name in fixed_dict:
                continue
            lower, upper, default, log = bounds_dict[name]
            hyper = CSH.UniformFloatHyperparameter("{}_{}".format(name, label),
                    lower=lower, upper=upper, default_value=default, log=log)
            hyperparameters.append(hyper)
        return hyperparameters

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        Q_hypers = self._get_hyperparameters("Q", self._Q_bounds, self._Q_fixed, self.system.observations)
        R_hypers = self._get_hyperparameters("R", self._R_bounds, self._R_fixed, self.system.controls)
        F_hypers = self._get_hyperparameters("F", self._F_bounds, self._F_fixed, self.system.observations)
        fixed_goal = set(self.system.observations)
        for obs_name in self._goal_tunable.keys():
            fixed_goal.remove(obs_name)
        goal_hypers = self._get_hyperparameters("Goal", self._goal_tunable, fixed_goal, self.system.observations)

        # Log Barrier
        log_hypers = self._get_hyperparameters_barrier("LogBarrier", self._scale_bounds, self._scale_fixed)
        cs.add_hyperparameters(Q_hypers + R_hypers + F_hypers + goal_hypers + log_hypers)
        return cs

    def is_compatible(self, ocp):
        if self.goal:
            return True
        else:
            return ocp.get_cost().has_goal

    def _get_matrix(self, cfg, label, fixed_dict, dim_names):
        mat = np.zeros((len(dim_names), len(dim_names)))
        for i, name in enumerate(dim_names):
            hyper_name = f"{name}_{label}"
            if name in fixed_dict:
                mat[i,i] = fixed_dict[name]
            elif hyper_name in cfg:
                mat[i,i] = cfg[hyper_name]
            else:
                mat[i,i] = 0.0
        return mat

    def _get_goal(self, cfg, ocp):
        if self.goal is None and ocp.get_cost().has_goal:
            goal = ocp.get_cost().get_goal()
        elif self.goal is not None: 
            goal = self.goal
        else:
            raise ValueError("QuadCostFactory requires goal")

        goal = np.nan_to_num(goal, nan=0.0)
        for obs_name in self._goal_tunable.keys():
            hyper_name = f"{obs_name}_Goal"
            goal[self.system.observations.index(obs_name)] = cfg[hyper_name]
        
        return goal

    def set_config(self, config):
        self.config = config
        
    def get_prototype(self, config, ocp):
        return PrototypeOCP(ocp, cost=QuadLimitCost)

    def __call__(self, ocp):
        Q = self._get_matrix(self.config, "Q", self._Q_fixed, self.system.observations)
        R = self._get_matrix(self.config, "R", self._R_fixed, self.system.controls)
        F = self._get_matrix(self.config, "F", self._F_fixed, self.system.observations)
        goal = self._get_goal(self.config, ocp)

        boundedStates = self._get_boundedState(self.config, "LogBarrier", self._scale_fixed)

        new_cost = QuadLimitCost(self.system, boundedStates, Q, R, F, goal=goal)
        new_ocp = copy.deepcopy(ocp)
        new_ocp.set_cost(new_cost)

        return new_ocp


    # Log Barrier Cost
    def set_limit(self, boundedState, limit, upper):
        if(boundedState in self.system.observations or boundedState in self.system.controls):
            self._limits[boundedState] = (limit, upper)
        else:
            raise ValueError(str(boundedState) + " is not in system")

    def set_bounds(self, boundedState, lower_bound, upper_bound, default, log=False):
        if(boundedState in self.system.observations or boundedState in self.system.controls):
            if(boundedState in self._limits):
                self._scale_bounds[boundedState] = (lower_bound, upper_bound, default, log)
            else:
                raise ValueError(str(boundedState) + " does not have a configured limit use set_limit")
        else:
            raise ValueError(str(boundedState) + " is not in system")

    def set_fixed_scale(self, boundedState, scale):
        if(boundedState in self.system.observations or boundedState in self.system.controls):
            if(boundedState in self._limits):
                self._scale_fixed[boundedState] = scale
            else:
                raise ValueError(str(boundedState) + " does not have a configured limit use set_limit")
        else:
            raise ValueError(str(boundedState) + " is not in system")

    def _get_boundedState(self, cfg, label, fixed_dict):
        boundedStates = dict()
        for name in (self.system.controls + self.system.observations):
            if name in fixed_dict:
                limit, upper = self._limits[name]
                scale = self._scale_fixed[name]
                boundedStates[name] = (limit, scale, upper)
            elif name in self._limits:
                limit, upper = self._limits[name]
                upper_string = "Upper"
                if(not upper):
                    upper_string = "Lower"
                hyper_name = f"{name}_{upper_string}_{label}"
                scale = cfg[hyper_name]
                boundedStates[name] = (limit, scale, upper)
        return boundedStates

    def _get_hyperparameters_barrier(self, label, bounds_dict, fixed_dict):
        hyperparameters = []
        for name in (self.system.controls + self.system.observations):
            if name in fixed_dict or name not in self._limits:
                continue
            limit, upper = self._limits[name]
            upper_string = "Upper"
            if(not upper):
                upper_string = "Lower"
            lower_scale, upper_scale, default, log = bounds_dict[name]
            hyper = CSH.UniformFloatHyperparameter("{}_{}_{}".format(name, upper_string, label),
                    lower=lower_scale, upper=upper_scale, default_value=default, log=log)
            hyperparameters.append(hyper)
        return hyperparameters
