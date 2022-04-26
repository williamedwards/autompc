import numpy as np

from .ocp import OCP, PrototypeOCP
from .ocp_factory import OCPFactory
from ..costs.sum_cost import SumCost
from ..utils.cs_utils import *

class SumFactory(OCPFactory):
    """
    A factory which produces sum of several cost terms. A SumFactory
    can be crated by combining other OCP factories with the `+` operator.
    All factories in the sum must produce the same output for observation
    and control bounds.
    """
    def __init__(self, system, factories):
        self._factories = factories
        super().__init__(system, "SumFactory")

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        for i, factory in enumerate(self._factories):
            _fact_cs = factory.get_config_space()
            add_configuration_space(cs,f"_sum_{i}", _fact_cs)
        return cs

    def is_compatible(self, ocp):
        for factory in self._factories:
            if not factory.is_compatible(ocp):
                return False
        return True

    def set_config(self, config):
        for i, factory in enumerate(self._factories):
            factory_config = create_subspace_configuration(config, f"_sum_{i}", 
                factory.get_config_space()) 
            factory.set_config(config)
        
    def get_prototype(self, config, ocp):
        prototypes = []
        for i, factory in enumerate(self._factories):
            factory_config = create_subspace_configuration(config, f"_sum_{i}", 
                factory.get_config_space()) 
            prototypes.append(factory.get_prototype(config, ocp))
        return SumPrototypeOCP(prototypes)

    def __call__(self, ocp):
        # Run all factories
        transformed_ocps = [factory(ocp) for factory in self._factories]

        # Check consistency of system, control/observation bounds
        system = transformed_ocps[0].system
        obs_bounds = transformed_ocps[0].get_obs_bounds()
        ctrl_bounds = transformed_ocps[0].get_ctrl_bounds()
        for transformed_ocp in transformed_ocps[1:]:
            if transformed_ocp.system != system:
                raise ValueError("All factories in SumFactory must use same system.")
            if not np.array_equal(transformed_ocp.get_obs_bounds(), obs_bounds):
                raise ValueError("All factories in SumFactory must produce same observation bounds.")
            if not np.array_equal(transformed_ocp.get_ctrl_bounds(), ctrl_bounds):
                raise ValueError("All factories in SumFactory must produce same control bounds.")

        # Construct SumCost
        sum_cost = SumCost(ocp.system, [ocp.get_cost() for ocp in transformed_ocps])

        # Construct output ocp
        new_ocp = transformed_ocps[0]
        new_ocp.set_cost(sum_cost)
            
        return new_ocp

    def __add__(self, other):
        if isinstance(other, SumFactory):
            return SumFactory(self.system, [*self._factories, *other._factories])
        else:
            return SumFactory(self.system, [*self._factories, other])

    def __radd__(self, other):
        if isinstance(other, SumFactory):
            return SumFactory(self.system, [*other.costs, *self.costs])
        else:
            return SumFactory(self.system, [other, *self._factories])


class SumPrototypeOCP(PrototypeOCP):
    def __init__(self, prototypes):
        # Check system match
        self.system = prototypes[0].system
        for prototype in prototypes[1:]:
            if prototype.system != self.system:
                raise RuntimeError("All factories in SumFactory must use same system.")

        # Check control bound agreement
        self.are_ctrl_bounded = prototypes[0].system
        for prototype in prototypes[1:]:
            if prototype.are_ctrl_bounded != self.are_ctrl_bounded:
                raise RuntimeError("All factories in SumFactory must use same control bounds.")

        # Check observation bound agreement
        self.are_obs_bounded = prototypes[0].system
        for prototype in prototypes[1:]:
            if prototype.are_obs_bounded != self.are_obs_bounded:
                raise RuntimeError("All factories in SumFactory must use same observation bounds.")

        self.cost = SumCost(system, [prototype.cost for prototype in prototypes])