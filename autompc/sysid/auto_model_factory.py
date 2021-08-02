
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC
from .model import Model, ModelFactory
from .mlp import MLPFactory
from .arx import ARXFactory
from ..utils.cs_utils import add_configuration_space

class AutoModelFactory(ModelFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factories = [MLPFactory(self.system), ARXFactory(self.system)]

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
            cfg = cs.get_default_configuration()
            prefix = "_" + factory.name + ":"
            for key, val in cfg_combined.get_dictionary().items():
                if key[:len(prefix)] == prefix:
                    cfg[key.split(":")[1]] = val
            return factory, cfg

    def __call__(self, cfg_combined, *args, **kwargs):
        factory, cfg = self._get_factory_cfg(cfg_combined)
        return factory(cfg, *args, **kwargs)

