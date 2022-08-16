# Created by William Edwards (wre2@illinois.edu)

from typing import List, Tuple, Optional
class System:
    """
    Stores constant information about controlled dynamical system, including
    the size of the control and observation dimensions and the labels for each
    control and observation variable.
    """
    def __init__(self, observations : List[str], controls : List[str], dt : Optional[float]=None):
        """
        Parameters
        ----------
        observations : List of strings
            Name of each observation (state) dimension

        controls : List of strings
            Name of each control dimension

        dt : float
            Optional: Data time step for the system.
        """
        # Check inputs
        obs_set = set(observations)
        ctrl_set = set(controls)
        e = ValueError("Observation and control labels must be unique")
        if len(obs_set) != len(observations):
            raise e
        if len(ctrl_set) != len(controls):
            raise e
        if ctrl_set.intersection(obs_set):
            raise e

        self._controls = controls[:]
        self._observations = observations[:]

        self._dt = dt

    def __eq__(self, other):
        return ((self.controls == other.controls) 
                and (self.observations == other.observations))
    
    def __str__(self):
        observation_str = '{} observations'.format(len(self._observations)) if len(self._observations) > 4 else '['+','.join(self._observations)+']'
        control_str = '{} controls'.format(len(self._controls)) if len(self._controls) > 4 else '['+','.join(self._controls)+']'
        if self._dt is None:
            return '{}({},{})'.format(self.__class__.__name__,observation_str,control_str)
        else:
            dt_str = "dt={:.3f}".format(self._dt)
            return '{}({},{},{})'.format(self.__class__.__name__,observation_str,control_str,dt_str)
            


    @property
    def controls(self) -> List[str]:
        """
        Names of each control dimension
        """
        return self._controls[:]

    @property
    def observations(self) -> List[str]:
        """
        Names of each observation dimension
        """
        return self._observations[:]

    @property
    def ctrl_dim(self) -> int:
        """
        Size of control dimensions
        """
        return len(self._controls)

    @property
    def obs_dim(self) -> int:
        """
        Size of observation dimensions
        """
        return len(self._observations)

    @property
    def dt(self) -> float:
        """
        Timestep.  Will be 1 if dt was not specified
        """
        return self._dt if self._dt is not None else 1
    
    @dt.setter
    def dt(self,dt : float):
        self._dt = dt

    @property
    def discrete_time(self) -> bool:
        """
        Whether dt was not specified on initialization
        """
        return self._dt is None