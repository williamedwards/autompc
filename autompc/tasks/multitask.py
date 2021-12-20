# Created by William Edwards (wre2@illinois.edu)

# Standard library includes

# External library includes
import numpy as np

# Project includes
from .task import Task

class MultiTask(Task):
    def __init__(self, system, subtasks):
        super().__init__(system)
        self._subtasks = subtasks[:]

    def get_subtasks(self):
        return self._subtasks[:]

    def add_subtask(self, task):
        self._subtasks.append(task)