
from .controller import ControllerFactory

class WrappedFactory(ControllerFactory):
    def __init__(self, controller_factory, WrapperClass):
        self.controller_factory = controller_factory
        self.WrapperClass = WrapperClass

    def get_configuration_space(self):
        return self.controller_factory.get_configuration_space()

    def __call__(self, *args, **kwargs):
        controller = self.controller_factory(*args, **kwargs)
        return self.WrapperClass(controller.system, controller.model, controller.task, controller)

    #def is_compatible