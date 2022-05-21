import numpy as np

from .ocp import OCP
from .ocp_transformer import OCPTransformer,PrototypeOCP
from ..costs.cost import SumCost
from ..utils.cs_utils import *

class SumTransformer(OCPTransformer):
    """
    A transformer which produces sum of several cost terms. A SumTransformer
    can be created by combining other OCP transformers with the `+` operator.
    All transformers in the sum must produce the same output for observation
    and control bounds.
    """
    def __init__(self, system, transformers):
        self._transformers = transformers
        super().__init__(system, "SumTransformer")

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        for i, transformer in enumerate(self._transformers):
            _fact_cs = transformer.get_config_space()
            add_configuration_space(cs,f"_sum_{i}", _fact_cs)
        return cs

    def is_compatible(self, ocp):
        for transformer in self._transformers:
            if not transformer.is_compatible(ocp):
                return False
        return True

    def ocp_requirements(self) -> dict:
        res = dict()
        for transformer in self._transformers:
            try:
                res.update(transformer.ocp_requirements())
            except NotImplementedError:
                pass
        return res

    def set_config(self, config):
        for i, transformer in enumerate(self._transformers):
            transformer_config = create_subspace_configuration(config, f"_sum_{i}", 
                transformer.get_config_space()) 
            transformer.set_config(transformer_config)
        
    def get_prototype(self, config, ocp):
        prototypes = []
        for i, transformer in enumerate(self._transformers):
            transformer_config = create_subspace_configuration(config, f"_sum_{i}", 
                transformer.get_config_space()) 
            prototypes.append(transformer.get_prototype(transformer_config, ocp))
        return SumPrototypeOCP(prototypes)

    def __call__(self, ocp):
        # Run all transformers
        transformed_ocps = [transformer(ocp) for transformer in self._transformers]

        # Check consistency of system, control/observation bounds
        system = transformed_ocps[0].system
        obs_bounds = transformed_ocps[0].get_obs_bounds()
        ctrl_bounds = transformed_ocps[0].get_ctrl_bounds()
        for transformed_ocp in transformed_ocps[1:]:
            if transformed_ocp.system != system:
                raise ValueError("All transformers in SumTransformer must use same system.")
            if not np.array_equal(transformed_ocp.get_obs_bounds(), obs_bounds):
                raise ValueError("All transformers in SumTransformer must produce same observation bounds.")
            if not np.array_equal(transformed_ocp.get_ctrl_bounds(), ctrl_bounds):
                raise ValueError("All transformers in SumTransformer must produce same control bounds.")

        # Construct SumCost
        sum_cost = SumCost(ocp.system, [ocp.get_cost() for ocp in transformed_ocps])

        # Construct output ocp
        new_ocp = transformed_ocps[0]
        new_ocp.set_cost(sum_cost)
            
        return new_ocp
    
    @property
    def trainable(self) -> bool:
        return any(transformer.trainable for transformer in self._transformers)

    def __add__(self, other):
        if isinstance(other, SumTransformer):
            return SumTransformer(self.system, [*self._transformers, *other._transformers])
        else:
            return SumTransformer(self.system, [*self._transformers, other])

    def __radd__(self, other):
        if isinstance(other, SumTransformer):
            return SumTransformer(self.system, [*other._transformers, *self._transformers])
        else:
            return SumTransformer(self.system, [other, *self._transformers])


class SumPrototypeOCP(PrototypeOCP):
    def __init__(self, prototypes):
        # Check system match
        self.system = prototypes[0].system
        for prototype in prototypes[1:]:
            if prototype.system != self.system:
                raise RuntimeError("All transformers in SumTransformer must use same system.")

        # Check control bound agreement
        self.are_ctrl_bounded = prototypes[0].system
        for prototype in prototypes[1:]:
            if prototype.are_ctrl_bounded != self.are_ctrl_bounded:
                raise RuntimeError("All transformers in SumTransformer must use same control bounds.")

        # Check observation bound agreement
        self.are_obs_bounded = prototypes[0].system
        for prototype in prototypes[1:]:
            if prototype.are_obs_bounded != self.are_obs_bounded:
                raise RuntimeError("All transformers in SumTransformer must use same observation bounds.")

        self.cost = SumCost(prototypes[0].system, [prototype.cost for prototype in prototypes])
