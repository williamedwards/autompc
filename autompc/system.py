# Created by William Edwards (wre2@illinois.edu)

class System:
    """
    The System object defines a robot system, including the size of the
    control and observation dimensions and the labels for each control
    and observation variable.
    """
    def __init__(self, observations, controls):
        """
        Parameters
        ----------
        observations : List of strings
            Name of each observation dimension

        controls : List of strings
            Name of each control dimension
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

    def __eq__(self, other):
        return ((self.controls == other.controls) 
                and (self.observations == other.observations))
    
    def __str__(self):
        observation_str = '{} observations'.format(len(self._observations)) if len(self._observations) > 4 else '['+','.join(self._observations)+']'
        control_str = '{} controls'.format(len(self._controls)) if len(self._controls) > 4 else '['+','.join(self._controls)+']'
        return '{}({},{})'.format(self.__class__.__name__,observation_str,control_str)

    @property
    def controls(self):
        """
        Names of each control dimension
        """
        return self._controls[:]

    @property
    def observations(self):
        """
        Names of each observation dimension
        """
        return self._observations[:]

    @property
    def ctrl_dim(self):
        """
        Size of control dimensions
        """
        return len(self._controls)

    @property
    def obs_dim(self):
        """
        Size of observation dimensions
        """
        return len(self._observations)
