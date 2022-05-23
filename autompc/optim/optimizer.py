# Created by William Edwards (wre2@illinois.edu)

from abc import abstractmethod
import copy
import numpy as np
from ..tunable import Tunable
from ..system import System
from ..sysid.model import Model
from ..ocp.ocp import OCP
from ..trajectory import Trajectory

class Optimizer(Tunable):
    def __init__(self, system : System, name : str):
        self.system = system
        self.name = name
        Tunable.__init__(self)

    @abstractmethod
    def step(self, obs : np.ndarray) -> np.ndarray:
        """
        Run the optimizer for a given time step

        Parameters
        ----------
            obs : numpy array of size self.system.obs_dim
                Current observation
        Returns
        -------
            ctrl : numpy array of size self.system.ctrl_dim
                Next control input
        """
        raise NotImplementedError

    def is_compatible(self, model : Model, ocp : OCP) -> bool:
        """
        Check if an optimizer is compatible with a given model
        and ocp.  Overridable if model_requirements, ocp_requirements, and
        cost_requirements aren't set.

        Parameters
        ----------
        model : Model
            Model to check compatibility.
        
        ocp : OCP
            OCP to check compatibility
        """
        model_reqs = self.model_requirements()
        ocp_reqs = self.ocp_requirements()
        cost_reqs = self.cost_requirements()
        for key,value in model_reqs.items():
            if getattr(model,key) != value:
                return False
        for key,value in ocp_reqs.items():
            if getattr(ocp,key) != value:
                return False
        cost = ocp.cost
        for key,value in cost_reqs.items():
            if getattr(cost,key) != value:
                return False
        return True
    
    def model_requirements(self) -> dict:
        """Returns a set of model properties that must hold for this optimizer
        to work.  For example `{'is_linear':True}` specifies that this optimizer
        requires a linear model.
        """
        raise NotImplementedError
    
    def ocp_requirements(self) -> dict:
        """Returns a set of ocp properties that must hold for this optimizer
        to work.  For example
        `{'are_obs_bounded':False,'are_ctrl_bounded':False}` specifies that
        this optimizer does not support bounds.
        """
        raise NotImplementedError

    def cost_requirements(self) -> dict:
        """Returns a set of cost properties that must hold for this optimizer
        to work.  For example `{'is_quad':True}` specifies that this
        optimizer only works with quadratic costs.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Re-initialize the optimizer. For optimizers which
        cache previous results to warm-start optimization, this
        will clear the cache.
        """
        pass

    def set_model(self, model : Model) -> None:
        """
        Set the model to be used for optimization.
        """
        self.model = model
    
    def set_ocp(self, ocp : OCP) -> None:
        """
        Set the OCP to be solved.
        """
        self.ocp = ocp

    def get_traj(self) -> Trajectory:
        """
        Returns the last optimized trajectory, if available.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Returns a representatation of the optimizers internal state.
        """
        raise NotImplementedError

    @abstractmethod
    def set_state(self, state):
        """
        Set the optimizers internal state.
        """
        raise NotImplementedError

    def clone(self) -> 'Optimizer':
        """
        Returns a deep copy of the optimizer.
        """
        return copy.deepcopy(self)