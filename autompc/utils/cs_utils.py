# Created by William Edwards (wre2@illinois.edu)

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC
import copy

from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    Hyperparameter,
    Constant,
    FloatHyperparameter,
    IntegerHyperparameter,
    NumericalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
)
from ConfigSpace.conditions import (
    ConditionComponent,
    AbstractCondition,
    AbstractConjunction,
    EqualsCondition,
)
from ConfigSpace.forbidden import (
    AbstractForbiddenComponent,
    AbstractForbiddenClause,
    AbstractForbiddenConjunction,
)

def _get_subkey(key, delimiter):
    return delimiter.join(key.split(delimiter)[1:])

def set_subspace_configuration(cfg, prefix, sub_cfg, delimiter=":"):
    prefix = prefix + delimiter
    for key, val in cfg.get_dictionary().items():
        if key.startswith(prefix):
            sub_cfg[_get_subkey(key, delimiter)] = val
        
def create_subspace_configuration(cfg, prefix, sub_cs, delimiter=":", **kwargs):
    prefix = prefix + delimiter
    values = dict()
    for key, val in cfg.get_dictionary().items():
        if key.startswith(prefix):
            values[_get_subkey(key, delimiter)] = val
    return Configuration(sub_cs, values=values, **kwargs)

def transfer_subspace_configuration(source_cfg, source_prefix, dest_cfg, dest_prefix,
        delimiter=":"):
    source_prefix = source_prefix + delimiter
    dest_prefix = dest_prefix + delimiter
    for source_key, source_val in source_cfg.get_dictionary().items():
        if source_key.startswith(source_prefix):
            key = _get_subkey(source_key, delimiter)
            dest_cfg[dest_prefix + key] = source_val

def set_parent_configuration(cfg, prefix, sub_cfg, delimiter=":"):
    prefix = prefix + delimiter
    for key, val in sub_cfg.get_dictionary().items():
        cfg[prefix+key] = val

def add_configuration_space(self,
                            prefix: str,
                            configuration_space: 'ConfigurationSpace',
                            delimiter: str = ":",
                            parent_hyperparameter: Hyperparameter = None
                            ):
    """
    Combine two configuration space by adding one the other configuration
    space. The contents of the configuration space, which should be added,
    are renamed to ``prefix`` + ``delimiter`` + old_name.

    Parameters
    ----------
    prefix : str
        The prefix for the renamed hyperparameter | conditions |
        forbidden clauses
    configuration_space : :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        The configuration space which should be added
    delimiter : str, optional
        Defaults to ':'
    parent_hyperparameter : :ref:`Hyperparameters`, optional
        Adds for each new hyperparameter the condition, that
        ``parent_hyperparameter`` is active

    Returns
    -------
    :class:`~ConfigSpace.configuration_space.ConfigurationSpace`
        The configuration space, which was added
    """
    if not isinstance(configuration_space, ConfigurationSpace):
        raise TypeError("The method add_configuration_space must be "
                        "called with an instance of "
                        "ConfigSpace.configuration_space."
                        "ConfigurationSpace.")

    new_parameters = []
    for hp in configuration_space.get_hyperparameters():
        new_parameter = copy.copy(hp)
        # Allow for an empty top-level parameter
        if new_parameter.name == '':
            new_parameter.name = prefix
        else:
            new_parameter.name = "%s%s%s" % (prefix, delimiter,
                                             new_parameter.name)
        self._add_hyperparameter(new_parameter)
        new_parameters.append(new_parameter)

    conditions_to_add = []
    for condition in configuration_space.get_conditions():
        new_condition = copy.copy(condition)
        dlcs = new_condition.get_descendant_literal_conditions()
        for dlc in dlcs:
            if dlc.child.name == prefix or dlc.child.name == '':
                dlc.child.name = prefix
            elif not dlc.child.name.startswith(
                            "%s%s" % (prefix, delimiter)):
                dlc_child_name = "%s%s%s" % (
                    prefix, delimiter, dlc.child.name)
                dlc.child = self.get_hyperparameter(dlc_child_name)
            if dlc.parent.name == prefix or dlc.parent.name == '':
                dlc.parent.name = prefix
            elif not dlc.parent.name.startswith(
                            "%s%s" % (prefix, delimiter)):
                dlc_parent_name = "%s%s%s" % (
                    prefix, delimiter, dlc.parent.name)
                dlc.parent = self.get_hyperparameter(dlc_parent_name)
        conditions_to_add.append(new_condition)
    self.add_conditions(conditions_to_add)

    if configuration_space.forbidden_clauses:
        raise ValueError("Forbidden clauses currently not supported by AutoMPC.")

    # forbiddens_to_add = []
    # for forbidden_clause in configuration_space.forbidden_clauses:
    #     # new_forbidden = copy.deepcopy(forbidden_clause)
    #     new_forbidden = forbidden_clause
    #     dlcs = new_forbidden.get_descendant_literal_clauses()
    #     for dlc in dlcs:
    #         if dlc.hyperparameter.name == prefix or \
    #                         dlc.hyperparameter.name == '':
    #             dlc.hyperparameter.name = prefix
    #         elif not dlc.hyperparameter.name.startswith(
    #                         "%s%s" % (prefix, delimiter)):
    #             dlc.hyperparameter.name = "%s%s%s" % \
    #                                       (prefix, delimiter,
    #                                        dlc.hyperparameter.name)
    #     forbiddens_to_add.append(new_forbidden)
    # self.add_forbidden_clauses(forbiddens_to_add)

    conditions_to_add = []
    if parent_hyperparameter is not None:
        for new_parameter in new_parameters:
            # Only add a condition if the parameter is a top-level
            # parameter of the new configuration space (this will be some
            #  kind of tree structure).
            if self.get_parents_of(new_parameter):
                continue
            condition = EqualsCondition(new_parameter,
                                        parent_hyperparameter['parent'],
                                        parent_hyperparameter['value'])
            conditions_to_add.append(condition)
    self.add_conditions(conditions_to_add)

def set_hyper_bounds(cs, hp_name, lower, upper, default_value=None, log=None):
    hp = cs.get_hyperparameter(hp_name)
    if not isinstance(hp, NumericalHyperparameter):
        raise ValueError("Can only call set_hyper_bounds for NumericalHyperparameter")
    name = hp.name
    if default_value is None:
        default_value = hp.default_value
    if log is None:
        log = hp.log
    if not (lower < default_value < upper):
        default_value = lower
    if isinstance(hp, UniformFloatHyperparameter):
        new_hp = CS.UniformFloatHyperparameter(name=name, lower=lower,
                upper=upper, default_value=default_value, log=log)
    if isinstance(hp, UniformIntegerHyperparameter):
        new_hp = CS.UniformIntegerHyperparameter(name=name, lower=lower,
                upper=upper, default_value=default_value, log=log)
    cs._hyperparameters[name] = new_hp

def set_hyper_choices(cs, hp_name, choices):
    hp = cs.get_hyperparameter(hp_name)
    if not isinstance(hp, CategoricalHyperparameter):
        raise ValueError("Can only call set_hyper_choices for CategoricalHyperparameter")
    name = hp.name
    default_value = hp.default_value
    if not default_value in choices:
        default_value = choices[0]
    new_hp = CS.CategoricalHyperparameter(name=name, choices=choices, default_value=default_value)
    cs._hyperparameters[name] = new_hp

def set_hyper_constant(cs, hp_name, value):
    hp = cs.get_hyperparameter(hp_name)
    name = hp.name
    new_hp = CS.Constant(name=name, value=value)
    cs._hyperparameters[name] = new_hp

def get_hyper_bool(config, name, default=False):
    try:
        val = config[name]
    except KeyError:
        return default
    if val is None:
        return default
    elif isinstance(val, bool):
        return val
    elif isinstance(val, str):
        if val == "true":
            return True
        elif val == "false":
            return False
        else:
            raise ValueError("Unknown value for boolean hyperparameter")
    else:
        raise ValueError("Unknown type for boolean hyperparameter")

def get_hyper_int(config, name, default=None):
    try:
        val = config[name]
    except KeyError:
        return default
    if val is None:
        return default
    elif isinstance(val, int):
        return val
    elif isinstance(val, str):
        return int(val)
    else:
        raise ValueError("Unknown type for integer hyperparameter")

def get_hyper_float(config, name, default=None):
    try:
        val = config[name]
    except KeyError:
        return default
    if val is None:
        return default
    elif isinstance(val, float):
        return val
    elif isinstance(val, str):
        return float(val)
    else:
        raise ValueError("Unknown type for float hyperparameter")

def get_hyper_str(config, name, default=None):
    try:
        val = config[name]
    except KeyError:
        return default
    if val is None:
        return default
    elif isinstance(val, str):
        return val
    else:
        raise ValueError("Unknown type for string hyperparameter")

def coerce_hyper_vals(cs, values):
    coerced_values = dict()

    for name, value in values.items():
        hyper = cs.get_hyperparameter(name)
        if isinstance(hyper, FloatHyperparameter):
            cvalue = float(value)
        if isinstance(hyper, IntegerHyperparameter):
            cvalue = int(value)
        if isinstance(hyper, CategoricalHyperparameter):
            if value in hyper.choices:
                cvalue = value
            elif str(value) in hyper.choices:
                cvalue = str(value)
            elif str(value).lower() in hyper.choices:
                cvalue = str(value).lower()
            else:
                cvalue = value
        coerced_values[name] = cvalue
    
    return coerced_values

def forbid_value(cs, name, value):
    """Safely forbids a value (not excluding it from a domain)"""
    #CS will complain if the default value is forbidden
    hyper = cs.get_hyperparameter(name)
    if hyper.default_value == value:
        hyper.default_value = [c for c in hyper.choices if c != value][0]
    match = CS.ForbiddenEqualsClause(hyper,value)
    cs.add_forbidden_clause(match)