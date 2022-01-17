

class OCPFactory:
    def __init__(self, system, ocp):
        self.system = system

    def get_config_space(self):
        return get_default_config_space()

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
