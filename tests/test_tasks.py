# Created by William Edwards (wre2@illinois.edu), 2021-12-19

# Standard library includes
import unittest
import copy

# Internal library includes
import autompc as ampc
from autompc.task import Task, StaticGoalTask
from autompc.sysid import SINDy
from autompc.costs import QuadCost, ThresholdCost
from autompc.optim import IterativeLQR

# External library includes
import numpy as np
import ConfigSpace as CS


def doubleint_dynamics(y, u):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    x, dx = y
    return np.array([dx, u])

def dt_doubleint_dynamics(y,u,dt):
    y += dt * doubleint_dynamics(y,u[0])
    return y

def uniform_random_generate(system, task, dynamics, rng, init_min, init_max, 
        traj_len, n_trajs):
    trajs = []
    for _ in range(n_trajs):
        state0 = [rng.uniform(minval, maxval, 1)[0] for minval, maxval 
                in zip(init_min, init_max)]
        y = state0[:]
        traj = ampc.zeros(system, traj_len)
        traj.obs[:] = y
        umin, umax = task.get_ctrl_bounds().T
        for i in range(traj_len):
            traj[i].obs[:] = y
            u = rng.uniform(umin, umax, 1)
            y = dynamics(y, u)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

class StaticGoalTaskTest(unittest.TestCase):
    def setUp(self):
        simple_sys = ampc.System(["x", "y"], ["u"])
        self.system = simple_sys

    def create_task(self):
        task = Task(self.system)
        task.set_ctrl_bound("u", -5, 10)
        cost = ThresholdCost(self.system, threshold=0.5)
        task.set_cost(cost)
        task.set_goal([2.0, 3.0])
        task.set_init_obs([0.0, -1.0])

        ret_cost = task.get_cost()
        cost_val1 = ret_cost.incremental([2.4, 3.1],[0])
        cost_val2 = ret_cost.incremental([4.4, 3.6],[0])

        task.set_goal([4.0, 4.0])
        ret_cost2 = task.get_cost()
        cost_val3 = ret_cost.incremental([2.4, 3.1],[0])
        cost_val4 = ret_cost.incremental([4.4, 3.6],[0])

        self.assertEqual(cost_val1, 0)
        self.assertEqual(cost_val2, 1)
        self.assertEqual(cost_val3, 1)
        self.assertEqual(cost_val4, 0)

        params = task.get_parameters()
        self.assertTrue((params["goal"] == np.array([4.0, 4.0])).all())
        self.assertTrue((params["init_obs"] == np.array([0.0, -1.0])).all())

    def test_mulititask(self):
        task = StaticGoalMultiTask(self.system)
        task.set_ctrl_bound("u", -5, 10)
        cost = ThresholdCost(self.system, threshold=0.5)
        task.set_cost(cost)
        task.set_default_parameter("init_obs", [0.0, -1.0])

        task.add_subtask(goal = np.array([4.0, 4.0]))
        task.add_subtask(goal = np.array([5.0, 6.0], 
            init_obs = np.array([1.0, 1.0])))
        
        params = list(task.get_subtask_parameters())
        self.assertTrue((params[0]["goal"] == np.array([4.0, 4.0])).all())
        self.assertTrue((params[0]["init_obs"] == np.array([0.0, -1.0])).all())
        self.assertTrue((params[1]["goal"] == np.array([5.0, 6.0])).all())
        self.assertTrue((params[1]["init_obs"] == np.array([1.0, 1.0])).all())

        subtasks = list(task.get_subtasks())
        self.assertTrue((subtasks[0].get_cost()._goal == np.array([4.0, 4.0])).all())
        self.assertTrue((subtasks[1].get_cost()._goal == np.array([5.0, 6.0])).all())
        self.assertTrue((subtasks[0].get_init_obs() == np.array([4.0, 4.0])).all())
        self.assertTrue((subtasks[1].get_init_obs() == np.array([5.0, 6.0])).all())


class UpdateTaskParametersTest(unittest.TestCase):
    def setUp(self):
        simple_sys = ampc.System(["x", "y"], ["u"])
        simple_sys.dt = 0.05
        self.system = simple_sys
        self.model_factory = SINDy(self.system)
        self.cost = QuadCost(self.system)
        self.optimizer = IterativeLQR(self.system)

        # Initialize task
        Q = np.eye(2)
        R = np.eye(1)
        F = np.eye(2)
        cost = QuadCost(self.system, Q, R, F)
        self.task = Task(self.system)
        self.task.set_cost(cost)
        self.task.set_ctrl_bound("u", -20.0, 20.0)
        self.task.set_goal([-1.0, 0.0])

        rng = np.random.default_rng(42)
        dynamics = lambda x, u: dt_doubleint_dynamics(x, u, dt=0.05)
        self.trajs = uniform_random_generate(self.system, self.task, 
                dynamics, rng, init_min=-np.ones(2), 
                init_max=np.ones(2), traj_len=100, n_trajs=100)
            
        self.pipeline = ampc.Pipeline(self.system, self.task, self.model_factory,
                self.cost_factory, self.controller_factory)
        pipeline_cs = self.pipeline.get_configuration_space()
        pipeline_cs.seed(100)
        pipeline_cfg = pipeline_cs.sample_configuration()
        self.controller, self.derived_task, self.model = self.pipeline(pipeline_cfg, self.trajs)
    
    def test_update_cost(self):
        cost = copy.deepcopy(self.derived_task.get_cost())
        self.assertTrue((cost._goal == np.array([-1.0, 0.0])))
        cost.update_task_parameters(goal=np.array([3.0, 10.0]))
        self.assertTrue((cost._goal == np.array([3.0, 10.0])))

    def test_update_controller(self):
        controller = copy.deepcopy(self.controller)
        self.assertTrue((controller.cost._goal == np.array([-1.0, 0.0])))
        controller.update_task_parameters(goal=np.array([3.0, 10.0]))
        self.assertTrue((controller.cost._goal == np.array([3.0, 10.0])))