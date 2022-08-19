from abc import ABC, abstractmethod
from collections import namedtuple
import time
import os
import numpy as np
from ..system import System
from ..dynamics import Dynamics
from ..trajectory import Trajectory
from ..policy import Policy
from ..controller import Controller
from ..task import Task
from ..utils import simulate
from typing import Union,List,Tuple,Callable
from joblib import Parallel, delayed
ControlEvaluationTrial = namedtuple("ControlEvaluationTrial", ["policy","task","dynamics","weight",
    "cost","traj","term_cond","eval_time"])

def trial_to_json(trial : ControlEvaluationTrial):
    res = trial._asdict()
    res['policy'] = str(trial.policy)
    res['task'] = str(trial.task)
    res['dynamics'] = str(trial.dynamics)
    res['traj'] = [trial.traj.obs.tolist(),trial.traj.ctrls.tolist()]
    return res

class ControlEvaluator(ABC):
    """An abstract method for evaluating a controller on one or more tasks.
    
    Arguments:
        system (System): the system
        tasks: the task or set of tasks to evaluate on.  Can also be a function
            which samples a task from a probability distribution.
    """
    def __init__(self, system : System, tasks : Union[Task,List[Task],Callable]):
        self.system = system
        if isinstance(tasks,Task):
            tasks = [tasks]
        self.tasks = tasks

    def __call__(self, policy : Union[Policy,Controller]) -> List[ControlEvaluationTrial]:
        """
        Evaluates policy on all tasks.  Default just runs evaluate_for_task
        on all tasks.
        
        Returns
        --------
            trial_info (List[ControlEvaluationTrial]):
                A list of trials evaluated.
        """
        if callable(self.tasks):
            raise ValueError("Can't use a task sampling function in evaluator class {}, must override __call__".format(self.__class__.__name__))
            
        results = []
        for i, task in enumerate(self.tasks):
            if hasattr(policy,'set_ocp'):  #it's a Controller
                policy.set_ocp(self.tasks[i])
            policy.reset()
            print(f"Evaluating Task {i}")
            trial_info = self.evaluate_for_task(policy, task)
            results.append(trial_info)
        return results

    @abstractmethod
    def evaluate_for_task(self, policy : Policy, task: Task) -> ControlEvaluationTrial:
        """
        Evaluates policy on one task. 
        
        Note: before calling this method, __call__ resets a controller for you
        and sets its optimal control problem to the task.
        
        Returns
        --------
            ControlEvaluationTrial
        """
        raise NotImplementedError
    
    def evaluate_for_task_dynamics(self, policy : Policy, task: Task, dynamics : Dynamics) -> ControlEvaluationTrial:
        """
        Standard evaluation of a policy on one task using a dynamics rollout.
        
        Note: before calling this method, __call__ resets a controller for you
        and sets its optimal control problem to the task.

        Note: you may use ._replace(attr=value) to change an attribute, e.g., weight.
        
        Returns
        --------
            ControlEvaluationTrial
        """
        t0 = time.time()
        try:
            truedyn_traj,truedyn_cost,truedyn_termcond = task.simulate(policy,dynamics)
            t1 = time.time()
            return ControlEvaluationTrial(policy=policy,task=task,dynamics=dynamics,weight=1.0,
                cost = truedyn_cost, traj=truedyn_traj, term_cond=truedyn_termcond, eval_time = t1-t0)
        except np.linalg.LinAlgError:
            truedyn_cost = np.inf
            t1 = time.time()
            return ControlEvaluationTrial(policy=policy,task=task,dynamics=dynamics,weight=1.0,
                cost = truedyn_cost, traj=truedyn_traj, term_cond='LinAlgError', eval_time = t1-t0)



class StandardEvaluator(ControlEvaluator):
    """A ControlEvaluator that is given a given dynamics model and just 
    simulates the controller on each task.

    Arguments:
        system (System): the system.
        tasks: the task or set of tasks to evaluate on.
        dynamics (Dynamics): the assumed dynamics for simulation.
    """
    def __init__(self, system, tasks, dynamics, prefix=''):
        super().__init__(system, tasks)
        self.dynamics = dynamics
        self.prefix = prefix

    def evaluate_for_task(self, controller : Policy, task : Task):
        print("Simulating Trajectory...",end='')
        controller.set_ocp(task)
        controller.reset()
        res = self.evaluate_for_task_dynamics(controller,task,self.dynamics)
        print("Resulting cost",res.cost)
        return res
