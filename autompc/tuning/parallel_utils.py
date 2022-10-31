# Created by William Edwards (wre2@illinois.edu), 2022-10-26

# Standard Library Includes
from typing import List, Callable, Any
from collections.abc import Iterable
from abc import ABC, abstractmethod

# External Library Includes
from joblib import Parallel, delayed
from dask.distributed import Client

class ParallelBackend(ABC):
    @abstractmethod
    def map(self, function: Callable[[Any], Any], values: Iterable) -> List[Any]:
        """
        Maps function and values in parallel.
        """
        raise NotImplementedError

class SerialBackend(ParallelBackend):
    def map(self, function: Callable[[Any], Any], values: Iterable) -> List[Any]:
        """ Implemented, see superclass. """
        return list(map(function, values))

class JoblibBackend(ParallelBackend):
    def __init__(self, n_jobs):
        self.n_jobs = n_jobs

    def map(self, function: Callable[[Any], Any], values: Iterable) -> List[Any]:
        """ Implemented, see superclass. """
        print("Entering JoblibBackend")
        return Parallel(n_jobs=self.n_jobs)(delayed(function)(value) for value in values)

class DaskBackend(ParallelBackend):
    def __init__(self, scheduler_address):
        self.scheduler_address = scheduler_address

    def map(self, function: Callable[[Any], Any], values: Iterable) -> List[Any]:
        """ Implemented, see superclass. """
        print("Entering DaskBackend")
        client = Client(self.scheduler_address)
        client.restart()
        futures = client.map(function, values)
        return [future.result() for future in futures]
