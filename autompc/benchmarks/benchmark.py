# Created by William Edwards (wre2@illinois.edu), 2021-01-08

# Standard library includes
from abc import ABC, abstractmethod

# External library includes
import numpy as np

class Benchmark(ABC):
    def __init__(self, name, system, task, data_gen_method):
        self.name = name
        self.system = system
        self.task = task
        self._data_gen_method = data_gen_method

    @abstractmethod
    def dynamics(self, x, u):
        raise NotImplementedError

    @abstractmethod
    def gen_trajs(self, seed):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def data_gen_methods(self):
        raise NotImplementedError
