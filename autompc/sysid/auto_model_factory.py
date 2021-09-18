
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC
from .model import Model, ModelFactory
from .mlp import MLPFactory
from .arx import ARXFactory
from .koopman import KoopmanFactory
from .sindy import SINDyFactory
from .largegp import ApproximateGPModelFactory
from ..utils.cs_utils import *

class AutoModelFactory(ModelFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factories = [MLPFactory(self.system), 
                ARXFactory(self.system),
                KoopmanFactory(self.system),
                SINDyFactory(self.system),
                ApproximateGPModelFactory(self.system)
                ]

    def get_configuration_space(self):
        cs_combined = CS.ConfigurationSpace()

        model_choice = CSH.CategoricalHyperparameter("model",
                choices=[factory.name for factory
                    in self.factories])
        cs_combined.add_hyperparameter(model_choice)
        for factory in self.factories:
            name = factory.name
            cs = factory.get_configuration_space()
            add_configuration_space(cs_combined, "_" + name, 
                    cs, parent_hyperparameter={"parent" : model_choice, 
                        "value" : name})

        return cs_combined

    def _get_factory_cfg(self, cfg_combined):
        for factory in self.factories:
            if factory.name != cfg_combined["model"]:
                continue
            cs = factory.get_configuration_space()
            prefix = "_" + factory.name
            cfg = create_subspace_configuration(cfg_combined, prefix, cs,
                allow_inactive_with_values=True)
            return factory, cfg

    def __call__(self, cfg_combined, *args, **kwargs):
        factory, cfg = self._get_factory_cfg(cfg_combined)
        return factory(cfg, *args, **kwargs)
