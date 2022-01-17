# Created by William Edwards (wre2@illinois.edu)

import copy
import numpy as np

class NumStepsTermCond:
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def __call__(self, traj):
        return len(traj) >= self.num_steps

class SingleTaskTuningProblem:
    def __init__(self, task):
        self._task = copy.deepcopy(task)

    def get_task(self):
        return copy.deepcopy(self._task)

    def set_task(self, task):
        self._task = copy.deepcopy(self._task)

class FiniteTasksTuningProblem:
    def __init__(self, tasks):
        self._tasks = copy.deepcopy(tasks)

    def get_tasks(self):
        return copy.deepcopy(self._tasks)

    def add_task(self, task):
        return self._tasks.append(copy.deepcopy(task))

class TaskDistriution:
    def __init__(self, sample_task):
        self.sample_task = sample_task

class Task:
    def __init__(self, system):
        """
        TuningProblem constructor

        Parameters
        ----------
        system : System
            System to be tuned
        """
        self.system = system

        self._term_cond = None
        self._num_steps = None
        self._ocp = None

    def set_ocp(self, ocp):
        """
        Sets the control problem.  This includes constriants
        and the performance metric.

        Parameters
        ----------
        ocp : ControlProblem
        """
        self._ocp = copy.deepcopy(ocp)

    def get_ocp(self, ocp):
        return copy.deepocpy(self._ocp)


    def set_num_steps(self, num_steps):
        """
        Sets maximum number of steps as the task terminiation
        condition.

        Parameters
        ----------
        num_steps : int
            Maximum number of steps.
        """
        self._term_cond = NumStepsTermCond(num_steps)
        self._num_steps = num_steps

    def has_num_steps(self):
        """
        Check whether task has a maximum number of steps for the
        task.

        Returns
        -------
        : bool
            True if maximum number of steps is set
        """
        return self._num_steps is not None

    def get_num_steps(self):
        """
        Returns the maxium number steps if available. None otherwise.
        """
        return self._num_steps

    def term_cond(self, traj):
        """
        Checks the task termination condition.

        Parameters
        ----------
        traj : Trajectory
            Trajectory to check termination condition.

        Returns
        -------
        : bool
            True if termination condition met.
        """
        if self._term_cond is not None:
            return self._term_cond(traj)
        else:
            return False

    def set_term_cond(self, term_cond):
        """
        Set the task termination condition

        Parameters
        ----------
        term_cond : Function, Trajectory -> bool
            Termination condition function.
        """
        self._term_cond = term_cond