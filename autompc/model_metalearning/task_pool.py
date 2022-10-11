from sys import stdout as _stdout
from mpi4py import MPI as _MPI
from time import time as _time
import logging

logging.basicConfig(
    level='INFO',
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

_sys_print = print


def print(*args, **kwargs):
    _sys_print(*args, **kwargs)
    _stdout.flush()


def _enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


class MPITaskPool:
    def __init__(self):
        self.start_time = None
        self.comm = _MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        self.status = _MPI.Status()
        if self.size < 2:
            raise Exception('At least 2 PEs are required, as one process will only manage ' +
                            'other processes.')

    def is_parent(self):
        return self.rank == 0

    def is_child(self):
        return self.rank != 0

    def run(self, tasks, func, log_freq=None):
        """
        A MPI-based implementation of the task-pool scheme of parallelism.
        Inputs:
        task     -- a list of tasks to be executed
        func     -- the function that will be applied to each task
        log_freq -- a message will be printed to STDOUT every `log_freq` tasks are finished.
                    If set to None, no log will be printed.
        Reference: https://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py
            (under MIT License)
        """
        self.start_time = _time()
        tags = _enum('READY', 'DONE', 'EXIT', 'START')

        # MASTER PROCESS
        if self.rank == 0:
            self.start_time = _time()
            num_workers = self.size - 1
            closed_workers = 0
            finished_tasks = 0
            itask = 0
            results = {}

            # distribution loop
            while closed_workers < num_workers:
                # wait for feedback from any process
                data = self.comm.recv(source=_MPI.ANY_SOURCE, tag=_MPI.ANY_TAG, status=self.status)
                src = self.status.Get_source()
                tag = self.status.Get_tag()

                # distribute new task
                if tag == tags.READY:
                    if itask < len(tasks):
                        task_buffer = (itask, tasks[itask])
                        self.comm.send(task_buffer, dest=src, tag=tags.START)
                        itask += 1
                    else:  # nothing more to be done
                        self.comm.send(None, dest=src, tag=tags.EXIT)

                # gather output
                elif tag == tags.DONE:
                    idx, res = data
                    results[idx] = res
                    finished_tasks += 1
                    if log_freq is not None and finished_tasks % log_freq == 0:
                        logging.info('%d out of %d tasks done at %.2f secs'
                                     % (finished_tasks, len(tasks), _time() - self.start_time))

                # exit condition
                elif tag == tags.EXIT:
                    closed_workers += 1

            self.comm.Barrier()
            results_sorted = [results[i] for i in range(len(tasks))]

        # WORKER PROCESSES
        else:
            while True:
                # wait for instruction from master process
                self.comm.send(None, dest=0, tag=tags.READY)
                task_buffer = self.comm.recv(source=0, tag=_MPI.ANY_TAG, status=self.status)
                tag = self.status.Get_tag()

                # work on task
                if tag == tags.START:
                    idx, task = task_buffer
                    res = func(task)
                    res_buffer = (idx, res)
                    self.comm.send(res_buffer, dest=0, tag=tags.DONE)

                # exit condition
                elif tag == tags.EXIT:
                    break

            self.comm.send(None, dest=0, tag=tags.EXIT)
            results_sorted = None
            self.comm.Barrier()

        return results_sorted