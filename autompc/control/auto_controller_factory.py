
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC
from .controller import Controller, ControllerFactory
from .lqr import LQRFactory
from .ilqr import IterativeLQRFactory
from .nmpc import DirectTranscriptionControllerFactory
from .mppi import MPPIFactory
from ..utils.cs_utils import add_configuration_space, create_subspace_configuration

class AutoControllerFactory(ControllerFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factories = [LQRFactory(self.system), 
                IterativeLQRFactory(self.system),
                DirectTranscriptionControllerFactory(self.system),
                MPPIFactory(self.system)
                ]


    def get_configuration_space(self):
        cs_combined = CS.ConfigurationSpace()

        controller_choice = CSH.CategoricalHyperparameter("controller",
                choices=[factory.name for factory
                    in self.factories])
        cs_combined.add_hyperparameter(controller_choice)
        for factory in self.factories:
            name = factory.name
            cs = factory.get_configuration_space()
            add_configuration_space(cs_combined, "_" + name, 
                    cs, parent_hyperparameter={"parent" : controller_choice, 
                        "value" : name})

        return cs_combined

    def _get_factory_cfg(self, cfg_combined):
        for factory in self.factories:
            if factory.name != cfg_combined["controller"]:
                continue
            cs = factory.get_configuration_space()
            prefix = "_" + factory.name
            cfg = create_subspace_configuration(cfg_combined, prefix, cs,
                allow_inactive_with_values=True)
            return factory, cfg

    def __call__(self, cfg_combined, *args, **kwargs):
        factory, cfg = self._get_factory_cfg(cfg_combined)
        return factory(cfg, *args, **kwargs)

