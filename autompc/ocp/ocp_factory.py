from abc import ABC, abstractmethod

class OCPFactory(ABC):
    def __init__(self, system, name):
        self.system = system
        self.name = name
        self.set_config(self.get_default_config())

    def get_config_space(self):
        return self.get_default_config_space()

    def get_default_config(self):
        return self.get_config_space().get_default_configuration()

    @abstractmethod
    def get_default_config_space(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, ocp):
        raise NotImplementedError

    @property
    def trainable(self):
        """
        Returns true for trainable models.
        """
        return not self.train.__func__ is OCPFactory.train

    @abstractmethod
    def get_prototype(self, config, ocp):
        """
        Returns a prototype of the output OCP for compatibility checking.
        """
        raise NotImplementedError