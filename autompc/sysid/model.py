# Created by William Edwards (wre2@illinois.edu)

# Standard library includes
from abc import abstractmethod
import copy
from ConfigSpace import Configuration

# External library includes
import numpy as np

# Internal library includes
from ..tunable import Tunable
from ..trajectory import Trajectory
from ..dynamics import Dynamics
from typing import List,Tuple,Any

class Model(Tunable,Dynamics):
    """A learnable model of a dynamic system.
    """
    def __init__(self, system, name):
        Tunable.__init__(self)
        Dynamics.__init__(self,system)
        self.name = name
        self.set_config(self.get_default_config())
        self.is_trained = False


    def get_obs(self,state):
        """Converts a state to an observation. Default assumes that
        observations are the first system.obs_dims dimensions.
        """
        return state[:self.system.obs_dim]

    def set_train_budget(self, seconds=None):
        """Sets the budget for training, in seconds."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clears all trained parameters.
        """
        raise NotImplementedError

    def clone(self) -> 'Model':
        """
        Returns a deep copy of the mdoel.
        """
        return copy.deepcopy(self)

    def to_linear(self) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Returns: (A, B, c)
            A, B -- Linear system matrices as Numpy arrays.
            c -- Linear system drift term as Numpy array.
        Only implemented for linear models.
        """
        raise NotImplementedError

    def train(self, trajs : List[Tuple], silent=False) -> None:
        """
        Parameters
        ----------
            trajs : List of pairs (xs, us)
                Training set of trajectories
            silent : bool
                Silence progress bar output

        Only implemented for trainable models. Subclasses should mark
        self.is_trained = True
        """
        raise NotImplementedError

    def get_parameters(self):
        """
        Returns a dict containing trained model parameters.

        Only implemented for trainable models.
        """
        raise NotImplementedError

    def set_parameters(self, params)  -> None:
        """
        Sets trainable model parameters from dict.

        Only implemented for trainable parameters.
        """
        raise NotImplementedError

    @property
    def is_linear(self) -> bool:
        """
        Returns true for linear models
        """
        return not self.to_linear.__func__ is Model.to_linear

    @property
    def is_diff(self) -> bool:
        """
        Returns true for differentiable models.
        """
        return not self.pred_diff.__func__ is Model.pred_diff

    @property
    def trainable(self) -> bool:
        """
        Returns true for trainable models.
        """
        return not self.train.__func__ is Model.train

    def get_prototype(self, config : Configuration) -> 'Model':
        """
        Returns a prototype of the model to be used for compatibility checking.
        It's only necessary to override this function when the compatibility
        properties depend on the config.
        """
        return self


class FullyObservableModel(Model):
    """A Model whose state = obs."""
    @property
    def state_dim(self):
        return self.system.obs_dim
    
    @property
    def state_system(self):
        return self.system

    def init_state(self, obs : np.ndarray) -> np.ndarray:
        return np.copy(obs)

    def traj_to_state(self, traj):
        return traj[-1].obs.copy()
    
    def update_state(self, state, new_ctrl, new_obs):
        return new_obs.copy()    

    def get_obs(self, state):
        return state