# Created by William Edwards (wre2@illinois.edu), 2021-01-24

# Standard library includes
import unittest
from pdb import set_trace

# Internal library includes
import autompc as ampc
from autompc.sysid import SINDyFactory, SINDy
from autompc.costs import QuadCostFactory, QuadCost, GaussRegFactory, SumCost
from autompc.tasks import Task
from autompc.control import IterativeLQRFactory, IterativeLQR
from autompc.pipeline import Pipeline

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

class PipelineTest(unittest.TestCase):
    def setUp(self):
        simple_sys = ampc.System(["x", "y"], ["u"])
        simple_sys.dt = 0.05
        self.system = simple_sys
        self.model_factory = SINDyFactory(self.system)
        self.cost_factory = QuadCostFactory(self.system)
        self.controller_factory = IterativeLQRFactory(self.system)

        # Initialize task
        Q = np.eye(2)
        R = np.eye(1)
        F = np.eye(2)
        cost = QuadCost(self.system, Q, R, F, goal=[-1,0])
        self.task = Task(self.system)
        self.task.set_cost(cost)
        self.task.set_ctrl_bound("u", -20.0, 20.0)

        rng = np.random.default_rng(42)
        dynamics = lambda x, u: dt_doubleint_dynamics(x, u, dt=0.05)
        self.trajs = uniform_random_generate(self.system, self.task, 
                dynamics, rng, init_min=-np.ones(2), 
                init_max=np.ones(2), traj_len=100, n_trajs=100)


    def test_full_config_space(self):
        pipeline = Pipeline(self.system, self.model_factory,
                self.cost_factory, self.controller_factory)
        pipeline_cs = pipeline.get_configuration_space()
        model_cs = self.model_factory.get_configuration_space()
        cost_cs = self.cost_factory.get_configuration_space()
        controller_cs = self.controller_factory.get_configuration_space()

        pipeline_hns = pipeline_cs.get_hyperparameter_names()
        combined_hns = (["_model:" + hn for hn in model_cs.get_hyperparameter_names()]
                      + ["_cost:" + hn for hn in cost_cs.get_hyperparameter_names()]
                      + ["_ctrlr:" + hn for hn in controller_cs.
                          get_hyperparameter_names()])

        self.assertEqual(set(pipeline_hns), set(combined_hns))

    def test_config_space_fixed_model(self):
        model_cs = self.model_factory.get_configuration_space()
        model_cfg = model_cs.get_default_configuration()
        model = self.model_factory(model_cfg, self.trajs)
        self.assertTrue(isinstance(model, SINDy))

        pipeline = Pipeline(self.system, model,
                self.cost_factory, self.controller_factory)
        pipeline_cs = pipeline.get_configuration_space()
        model_cs = self.model_factory.get_configuration_space()
        cost_cs = self.cost_factory.get_configuration_space()
        controller_cs = self.controller_factory.get_configuration_space()

        pipeline_hns = pipeline_cs.get_hyperparameter_names()
        combined_hns = (["_cost:" + hn for hn in cost_cs.get_hyperparameter_names()]
                      + ["_ctrlr:" + hn for hn in controller_cs.
                          get_hyperparameter_names()])

        self.assertEqual(set(pipeline_hns), set(combined_hns))

    def test_config_space_fixed_cost(self):
        cost = self.task.get_cost()

        pipeline = Pipeline(self.system, self.model_factory,
                cost, self.controller_factory)
        pipeline_cs = pipeline.get_configuration_space()
        model_cs = self.model_factory.get_configuration_space()
        cost_cs = self.cost_factory.get_configuration_space()
        controller_cs = self.controller_factory.get_configuration_space()

        pipeline_hns = pipeline_cs.get_hyperparameter_names()
        combined_hns = (["_model:" + hn for hn in model_cs.get_hyperparameter_names()]
                      + ["_ctrlr:" + hn for hn in controller_cs.
                          get_hyperparameter_names()])

        self.assertEqual(set(pipeline_hns), set(combined_hns))

    def test_pipeline_call(self):
        pipeline = Pipeline(self.system, self.model_factory,
                self.cost_factory, self.controller_factory)
        pipeline_cs = pipeline.get_configuration_space()
        pipeline_cfg = pipeline_cs.get_default_configuration()
        controller, task, model = pipeline(pipeline_cfg, self.task, self.trajs)

        self.assertIsInstance(controller, IterativeLQR)
        self.assertIsInstance(task, Task)
        self.assertIsInstance(model, SINDy)
