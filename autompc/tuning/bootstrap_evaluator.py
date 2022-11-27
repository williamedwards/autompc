import numpy as np
import os
from typing import List, Optional
from functools import partial

from joblib import Parallel, delayed

from .control_evaluator import StandardEvaluator
from .parallel_evaluator import ParallelEvaluator
from .parallel_utils import ParallelBackend, JoblibBackend
from ..system import System
from ..trajectory import Trajectory
from ..sysid.model import Model
from .data_store import DataStore

def _train_bootstrap(bootstrap_sample: List[Trajectory], base_surrogate: Model, data_store: Optional[DataStore] = None):
    bootstrap_surrogate = base_surrogate.clone()
    bootstrap_surrogate.train(bootstrap_sample)
    if data_store:
        bootstrap_surrogate = data_store.wrap(bootstrap_surrogate)
    return bootstrap_surrogate

class BootstrapSurrogateEvaluator(ParallelEvaluator):
    """A surrogate evaluator that samples n_bootstraps data draws to train an
    ensemble of surrogate models.
    """
    def __init__(self, system : System, tasks, surrogate : Model, trajs : List[Trajectory], 
                 n_bootstraps=10, surrogate_tune_iters=100, rng=None, backend=None, data_store=None):
        if rng is None:
            rng = np.random.default_rng()
        
        #perform bootstrap sample to create surrogate ensemble
        population = np.empty(len(trajs), dtype=object)
        for i, traj in enumerate(trajs):
            population[i] = traj
        bootstrap_samples = []
        for i in range(n_bootstraps):
            bootstrap_samples.append(rng.choice(population, len(trajs), replace=True, axis=0))
        # surrogate_dynamics = Parallel(n_jobs=n_bootstraps)(delayed(_train_bootstrap)(surrogate, bootstrap_sample, data_store)
        #     for bootstrap_sample in bootstrap_samples)
        if backend is None:
            backend = JoblibBackend(n_jobs=os.cpu_count())
        surrogate_dynamics = backend.map(partial(_train_bootstrap, base_surrogate=surrogate, data_store=data_store), bootstrap_samples)

        ParallelEvaluator.__init__(self, StandardEvaluator(system,tasks,None,'surr_',data_store), surrogate_dynamics, backend=backend, data_store=data_store)
