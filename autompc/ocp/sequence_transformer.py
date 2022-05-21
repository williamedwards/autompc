import numpy as np

from .ocp import OCP
from .ocp_transformer import OCPTransformer,PrototypeOCP
from ..utils.cs_utils import *

class SequenceTransformer(OCPTransformer):
    """
    A transformer which performs a sequence of transformations. If transformers=
    [t1,t2,t3], then::

        result = t1 ( t2 ( t3 (ocp)))

    """
    def __init__(self, system, transformers):
        self._transformers = transformers
        super().__init__(system, "SequenceTransformer")

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        for i, transformer in enumerate(self._transformers):
            _fact_cs = transformer.get_config_space()
            add_configuration_space(cs,f"_seq_{i}", _fact_cs)
        return cs

    def is_compatible(self, ocp):
        for transformer in self._transformers:
            if not transformer.is_compatible(ocp):
                return False
        return True
        
    def ocp_requirements(self) -> dict:
        return self._transformers[0].ocp_requirements()

    def set_config(self, config):
        for i, transformer in enumerate(self._transformers):
            transformer_config = create_subspace_configuration(config, f"_seq_{i}", 
                transformer.get_config_space()) 
            transformer.set_config(transformer_config)
        
    def get_prototype(self, config, ocp):
        # Run all transformers in reverse order
        for i, transformer in reversed(list(enumerate(self._transformers))):
            transformer_config = create_subspace_configuration(config, f"_sum_{i}", 
                transformer.get_config_space()) 
            ocp = transformer.get_prototype(transformer_config, ocp)
        return ocp

    def __call__(self, ocp):
        # Run all transformers in reverse order
        for i, transformer in reversed(list(enumerate(self._transformers))):
            ocp = transformer(ocp)
        return ocp

    @property
    def trainable(self) -> bool:
        return any(transformer.trainable for transformer in self._transformers)

    def __matmul__(self, other):
        if isinstance(other, SequenceTransformer):
            return SequenceTransformer(self.system, [*self._transformers, *other._transformers])
        else:
            return SequenceTransformer(self.system, [*self._transformers, other])

    def __rmatmul__(self, other):
        if isinstance(other, SequenceTransformer):
            return SequenceTransformer(self.system, [*other._transformers, *self._transformers])
        else:
            return SequenceTransformer(self.system, [other, *self._transformers])

