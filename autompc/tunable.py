from abc import ABC, abstractmethod
import warnings
from typing import List,Optional
from ConfigSpace import Configuration,ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
import ConfigSpace as CS


class Tunable(ABC):
    """An abstract base class for a tunable component.  Implementers
    of subclasses will define the default configuration space, and
    optionally bias the hyperparameter distributions.

    Pre-tuning, users can modify the configuration space by changing 
    bounds, fixing values, etc.

    Post-tuning, users can also modify the configuration using
    `set_hyper_values`.

    To mark that you are done tuning a Tunable instance, you can call
    `freeze_hyperparameters()`. You may also check `is_tunable()` to see
    if an item has no hyperparameters or has been frozen.
    """
    def __init__(self):
        self._configuration_space = self.get_default_config_space()
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
        for key,value in kwargs.items():
            config[key] = value
        self.set_config(config)

    def fix_hyperparameters(self,**kwargs) -> None:
        """Freezes hyperparameter values in the configuration space
        AND in the current configuration. Hyperparameters are given
        as keyword arguments. 
        """
        config = self.get_config()
        for key,value in kwargs.items():
            hyperparam = self._configuration_space.get_hyperparameters_dict()[key]
            if isinstance(hyperparam,CSH.CategoricalHyperparameter):
                hyperparam.choices = (value,)
            else:
                self._configuration_space._hyperparameters[key] = CSH.Constant(key, value)
            hyperparam.default_value = value
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

    def set_hyperparameter_logs(self,**kwargs) -> None:
        """Sets the configuration space log values for a set of
        hyperparameters, as keyword arguments.
        """
        hyperparams = self._configuration_space.get_hyperparameters_dict()
        for key,log in kwargs.items():
            hyperparams[key].log = log

    def freeze_hyperparameters(self):
        """Denotes that this instance should no longer be tunable."""
        self._configuration_space = ConfigurationSpace()

    def is_tunable(self):
        """Denotes that this instance is not tunable"""
        return len(self._configuration_space.get_hyperparameters_dict())>=0


class NonTunable(Tunable):
    """A class compatible with Tunable but with no tunable parameters."""
    def get_default_config_space(self):
        return ConfigurationSpace()


def _add_conditional_config_space(cs, label, choices) -> None:
    from .utils.cs_utils import add_configuration_space

    if not choices: return
    choice_hyper = CS.CategoricalHyperparameter(label,
        choices=[choice.name for choice in choices])
    cs.add_hyperparameter(choice_hyper)
    for choice in choices:
        add_configuration_space(cs, choice.name,
            choice.get_config_space(),
            parent_hyperparameter={"parent" : choice_hyper,
                "value" : choice.name}
            )

def _get_choice_by_name(choices, name) -> Optional[Tunable]:
    for choice in choices:
        if choice.name == name:
            return choice
    return None

class TunablePipeline:
    """A pipeline including one or more tunable components.  The hyperparameter space
    is the Cartesian product of all components' hyperparameter spaces.

    To use::
        pipeline.add_component('A',[A_option1,A_option2])
        pipeline.add_component('B',[B_option1,B_option2])
    """
    def __init__(self):
        self._components = dict()
        self._component_order = []
        self._config = None
        self._fixed = dict()
        self._forbidden = set()

    def add_component(self, name: str, options : List[Tunable]):
        if name in self._components:
            raise ValueError("Can't duplicate a component")
        self._components[name] = None
        self.set_component(name,options)
        self._component_order.append(name)

    def set_component(self, name: str, options : List[Tunable]):
        if name not in self._components:
            raise ValueError("{} is not an added component".format(name))
        for opt in options:
            if not isinstance(opt,Tunable):
                raise ValueError("Item must be a tunable, got {} instead",opt.__class__.__name__)
            if not hasattr(opt,'name'):
                raise ValueError("Cant add a tunable that doesn't have a 'name' attribute")
        self._components[name] = options[:]
        
    def add_component_option(self, component : str, option : Tunable):
        if not isinstance(option,Tunable):
            raise ValueError("Item must be a tunable, got {} instead",option.__class__.__name__)
        if not hasattr(option,'name'):
            raise ValueError("Cant add a tunable that doesn't have a 'name' attribute")
        self._components[component].append(option)
    
    def fix_option(self, component : str, option : str):
        """Restricts component to take on value option."""
        for opt in self._components[component]:
            if opt.name == option:
                self._fixed[component]=option
                if self._config is not None:
                    self._config[component] = option
                return
        raise ValueError("Invalid option {} for component {}".format(option,component))
    
    def free_option(self,component):
        """Releases all previously set fix_option / forbid_option constraints
        on the given component.
        """
        if component in self._fixed:
            del self._fixed[component]
        remove = []
        for (c1,o1,c2,o2) in self._forbidden:
            if c2 is None and c1 == component:
                remove.append((c1,o1,c2,o2))
        for r in remove:
            self._forbidden.remove(r)
    
    def get_config_space(self) -> ConfigurationSpace:
        """
        Returns the joint pipeline configuration space.
        """
        from .utils.cs_utils import forbid_value

        cs = ConfigurationSpace()
        for key,opts in self._components.items():
            _add_conditional_config_space(cs, key, opts)
        
        hyperparameters = cs.get_hyperparameters_dict()
        for (component,option) in self._fixed.items():
            hyperparam = hyperparameters[component]
            assert isinstance(hyperparam,CSH.CategoricalHyperparameter)
            hyperparam.choices = (option,)
            hyperparam.default_value = option
        for (component1,option1,component2,option2) in self._forbidden:
            if component2 is None:
                forbid_value(cs,component1,option1)
                continue  #requesting component1!=option1
            if len(self._components[component1]) == 1:
                forbid_value(cs,component2,option2)
                continue #no way for component2=option2 to be used
            if len(self._components[component2]) == 1:
                forbid_value(cs,component1,option1)
                continue #no way for component1=option1 to be used
            match1 = CS.ForbiddenEqualsClause(hyperparameters[component1],option1)
            match2 = CS.ForbiddenEqualsClause(hyperparameters[component2],option2)
            conjunction = CS.ForbiddenAndConjunction(match1,match2)
            cs.add_forbidden_clause(conjunction)
        return cs
    
    def set_config(self, config):
        from .utils.cs_utils import create_subspace_configuration
        self._config = config
        for name in self._component_order:
            opt = _get_choice_by_name(self._components[name], config[name])
            if opt is None:
                continue
            opt_config = create_subspace_configuration(config,opt.name,opt.get_config_space())
            opt.set_config(opt_config)
        
    def get_config(self):
        return self._config

    def get_configured_pipeline(self):
        """Call this after set_config() to get the pipeline."""
        pipeline = []
        for name in self._component_order:
            opt = _get_choice_by_name(self._components[name], self._config[name])
            pipeline.append(opt)
        return pipeline

    def get_default_config(self) -> Configuration:
        """
        Returns the default controller configuration.
        """
        return self.get_config_space().get_default_configuration()

    def set_hyper_values(self, component : str, option : str = None, **kwargs) -> None:
        """
        Set component.option hyperparameters by keyword argument. Also, fixes the component
        to use the given option.

        Parameters
        ---------- 
            component : str
                Name of component
            option : str
                Name of option for component
            **kwargs
                Model hyperparameter values
        """
        options = self._components[component]
        if option is None and len(options) == 1:
            option = options[0].name
        elif option is None:
            raise ValueError("Multiple options ({}) are present so name must be specified".format(', '.join(opt.name for opt in options)))
        option = _get_choice_by_name(options, option)
        if option is None:
            raise ValueError("Unrecognized model name")
        option.set_hyper_values(**kwargs)
        self.fix_option(component,option.name)

    def forbid_option(self, component, option):
        """Forbids the configuration space from choosing component = option."""
        if component not in self._components:
            raise ValueError("Invalid component specified")
        self._forbidden.add((component,option,None,None))

    def forbid_incompatible_options(self,component1 : str, option1 : str, component2 : str, option2 : str) -> None:
        """Forbids the configuration space from exploring the conjunction of
        component1=option1 and component2=option2.
        """
        if component1 not in self._components:
            raise ValueError("Invalid component specified")
        if component2 not in self._components:
            raise ValueError("Invalid component specified")
        self._forbidden.add((component1,option1,component2,option2))
