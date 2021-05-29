# Created by William Edwards (wre2@illinois.edu)

class Grapher:
    def __init__(self, system):
        self.system = system

    def __call__(self, model, configuration):
        """
        Returns an instantiated Graph instance.
        """
        raise NotImplementedError
