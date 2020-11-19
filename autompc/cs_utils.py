# Created by William Edwards (wre2@illinois.edu)

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC
import copy

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    Hyperparameter,
    Constant,
    FloatHyperparameter,
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

def set_subspace_configuration(cfg, prefix, sub_cfg, delimiter=":"):
    prefix = prefix + delimiter
    for key, val in cfg.get_dictionary().items():
        if key[:len(prefix)] == prefix:
            sub_cfg[key.split(delimiter)[1]] = val

def transfer_subspace_configuration(source_cfg, source_prefix, dest_cfg, dest_prefix,
        delimiter=":"):
    source_prefix = source_prefix + delimiter
    dest_prefix = dest_prefix + delimiter
    for source_key, source_val in source_cfg.get_dictionary().items():
        if source_key[:len(source_prefix)] == source_prefix:
            key = source_key.split(delimiter)[1]
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
        new_parameters.append(new_parameter)
    self.add_hyperparameters(new_parameters)

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

    forbiddens_to_add = []
    for forbidden_clause in configuration_space.forbidden_clauses:
        # new_forbidden = copy.deepcopy(forbidden_clause)
        new_forbidden = forbidden_clause
        dlcs = new_forbidden.get_descendant_literal_clauses()
        for dlc in dlcs:
            if dlc.hyperparameter.name == prefix or \
                            dlc.hyperparameter.name == '':
                dlc.hyperparameter.name = prefix
            elif not dlc.hyperparameter.name.startswith(
                            "%s%s" % (prefix, delimiter)):
                dlc.hyperparameter.name = "%s%s%s" % \
                                          (prefix, delimiter,
                                           dlc.hyperparameter.name)
        forbiddens_to_add.append(new_forbidden)
    self.add_forbidden_clauses(forbiddens_to_add)

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

def set_hyper_bounds(cs, hp_name, lower, upper):
    hp = cs.get_hyperparameter(hp_name)
    if not isinstance(hp, NumericalHyperparameter):
        raise ValueError("Can only call set_hyper_bounds for NumericalHyperparameter")
    name = hp.name
    default_value = hp.default_value
    if not (lower < default_value < upper):
        default_value = lower
    if isinstance(hp, UniformFloatHyperparameter):
        new_hp = CS.UniformFloatHyperparameter(name=name, lower=lower,
                upper=upper, default_value=default_value)
    if isinstance(hp, UniformIntegerHyperparameter):
        new_hp = CS.UniformIntegerHyperparameter(name=name, lower=lower,
                upper=upper, default_value=default_value)
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
