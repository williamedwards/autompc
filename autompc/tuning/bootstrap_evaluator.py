import numpy as np
from typing import List

from .control_evaluator import StandardEvaluator
from .parallel_evaluator import ParallelEvaluator
from ..system import System
from ..trajectory import Trajectory
from ..sysid.model import Model

class BootstrapSurrogateEvaluator(ParallelEvaluator):
    """A surrogate evaluator that samples n_bootstraps data draws to train an
    ensemble of surrogate models.
    """
    def __init__(self, system : System, tasks, trajs : List[Trajectory], surrogate : Model, n_bootstraps=10, surrogate_tune_iters=100, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        #perform bootstrap sample to create surrogate ensemble
        surrogate_dynamics = [surrogate.clone() for i in range(n_bootstraps)]
        for i in range(n_bootstraps):
            bootstrap_sample = rng.choice(trajs, len(trajs), replace=True, axis=0)
            surrogate_dynamics[i].train(bootstrap_sample)

        ParallelEvaluator.__init__(self, StandardEvaluator(system,tasks,None,'surr_'), surrogate_dynamics, len(surrogate_dynamics))
        

    
