from abc import ABC, abstractmethod

class OCPFactory(ABC):
    def __init__(self, system, name):
        self.system = system
        self.name = name

    def get_config_space(self):
        return self.get_default_config_space()

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
