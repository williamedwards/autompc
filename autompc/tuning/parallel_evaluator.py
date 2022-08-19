import numpy as np
from joblib import Parallel, delayed
from typing import Union,List,Tuple
from collections import defaultdict

from .control_evaluator import ControlEvaluator,StandardEvaluator,ControlEvaluationTrial
from ..task import Task
from ..policy import Policy
from ..dynamics import Dynamics

class ParallelEvaluator(ControlEvaluator):
    """A ControlEvaluator that runs one or more dynamics models and one or more
    tasks and performs them in parallel
    """
    def __init__(self, evaluator : StandardEvaluator,
                dynamics : Union[Dynamics,List[Dynamics]],
                max_jobs=1):
        if isinstance(evaluator,ParallelEvaluator):
            raise ValueError("Can't pass ParallelEvaluator to ParallelEvaluator constructor")
        super().__init__(evaluator.system, evaluator.tasks)
        self.evaluator = evaluator
        if not hasattr(dynamics,'__iter__'):
            dynamics = [dynamics]
        self.dynamics_models = dynamics
        self.max_jobs = max_jobs
    
    def num_jobs(self) -> int:
        return min(self.max_jobs,len(self.dynamics_models)*len(self.tasks))

    def run_job(self, controller, job_idx) -> ControlEvaluationTrial:
        task_idx, model_idx = job_idx % len(self.tasks), job_idx // len(self.tasks)
        surrogate = self.dynamics_models[model_idx]
        print("Simulating Surrogate Trajectory for Model {}, Task {}: ".format(model_idx, task_idx))
        self.evaluator.dynamics = surrogate
        result = self.evaluator.evaluate_for_task(controller, self.tasks[task_idx])
        return result

    def __call__(self, controller : Policy):
        if self.max_jobs > 1:
            print("Entering Parallel Evaluation")
            if hasattr(controller,'model') and hasattr(controller.model,'set_device'):
                controller.model.set_device("cpu")
            for dyn in self.dynamics_models:
                if hasattr(dyn,'set_device'):
                    dyn.set_device("cpu")
            results = Parallel(n_jobs=self.max_jobs)(delayed(self.run_job)(controller, i) 
                for i in range(self.num_jobs()))
            return results
        else:
            print("Entering Serial Evaluation")
            results = [self.run_job(controller, i) for i in range(self.num_jobs())]
            return results
    
    def evaluate_for_task(self, controller : Policy, task : Task):
        old_tasks = self.tasks
        self.tasks = [task]
        try:
            res = self(controller)
        except Exception:
            print("Error evaluating for single task")
        finally:
            self.tasks = old_tasks
        return res