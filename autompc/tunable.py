from abc import ABC, abstractmethod
import warnings
from ConfigSpace import Configuration,ConfigurationSpace

class Tunable(ABC):
    """An abstract base class for a tunable component.  Implementers
    of subclasses will define the default configuration space, and
    optionally 

    Pre-tuning, users can modify the configuration space by changing 
    bounds, fixing values, etc.

    Post-tuning, users can also modify the configuration using
    `set_hyper_values`.
    """
    def __init__(self):
        self._configuration_space = self.get_default_config_space()
        self._fixed_configuration_space_parameters = dict() # Key: parameter_value, Value: fixed_value
        self._config = self.get_default_config()
        self.set_config(self._config)

    def get_config_space(self) -> ConfigurationSpace:
        """
        Returns the model configuration space.
        """
        return self._configuration_space

    @abstractmethod
    def get_default_config_space(self) -> ConfigurationSpace:
        """
        Returns the default configuration space, prior to any user modifications.
        """
        raise NotImplementedError

    def get_default_config(self) -> Configuration:
        """
        Returns the default configuration.
        """
        return self.get_config_space().get_default_configuration()

    def set_config(self, config : Configuration) ->None:
        """
        Set the current configuration. Overridable. Default
        just sets the current config which can be retrieved
        via `get_config()`.

        Parameters
        ----------
            config : Configuration
                Configuration to set.
        """
        self._config = config
    
    def get_config(self) -> Configuration:
        """Retrieves the current configuration."""
        return self._config

    def set_hyper_values(self, **kwargs) -> None:
        """
        Set hyperparameter values in the current configuration,
        given as keyword arguments.
        """
        config = self.get_config()
        config.update(kwargs)
        self.set_config(config)

    def fix_hyperparameters(self,**kwargs) -> None:
        """Freezes hyperparameter values in the configuration space
        AND in the current configuration. Hyperparameters are given
        as keyword arguments. 
        """
        retained_hyperparameters = None
        for (key,value)  in kwargs.items():
            if key not in self._fixed_configuration_space_parameters:
                #need to remove from the configuration space
                if retained_hyperparameters is None:
                    retained_hyperparameters = self._configuration_space.get_hyperparameters_dict()
                del retained_hyperparameters[key]
        if retained_hyperparameters is not None:
            if self._configuration_space.get_conditions():
                warnings.warn("TODO: fix hyperparameters with conditions?")
            if self._configuration_space.get_forbiddens():
                warnings.warn("TODO: fix hyperparameters with conditions?")
            new_config_space = ConfigurationSpace(self._configuration_space.name,self._configuration_space.seed,self._configuration_space.meta)
            for (key,param) in retained_hyperparameters.items():
                new_config_space.add_hyperparameter(param)
            self._configuration_space = new_config_space

        config = self.get_config()
        for (key,value)  in kwargs.items():
            self._fixed_configuration_space_parameters[key] = value
            config[key] = value
        self.set_config(config)    
    
    def set_hyperparameter_bounds(self,**kwargs) -> None:
        """Sets the configuration space bounds for a set of hyperparameters,
        given as keyword arguments mapping to lower/upper pairs. 
        
        For example, `set_hyper_parameter_bounds(horizon=(3,10))` sets the
        horizon lower value to 3 and the upper value to 10.  A value of None
        in either the first or second element keeps the same value.
        """
        hyperparams = self._configuration_space.get_hyperparameters_dict()
        for key,bounds in kwargs.items():
            if len(bounds) != 2: raise ValueError("Bounds must be given as lower/upper pairs.")
            lower,upper = bounds
            if lower is not None:
                hyperparams[key].lower = lower
            if upper is not None:
                hyperparams[key].upper = upper
    
    def set_hyperparameter_defaults(self,**kwargs) -> None:
        """Sets the configuration space default values for a set of
        hyperparameters, as keyword arguments.
        """
        hyperparams = self._configuration_space.get_hyperparameters_dict()
        for key,defaults in kwargs.items():
            hyperparams[key].default_value = defaults
        
        