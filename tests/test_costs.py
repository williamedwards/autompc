# Created by William Edwards (wre2@illinois.edu), 2021-01-24

# Standard library includes
import unittest

# Internal library includes
import autompc as ampc
from autompc.sysid import ARX
from autompc.costs import QuadCostFactory, QuadCost
from autompc.tasks import Task

# External library includes
import numpy as np
import ConfigSpace as CS


class QuadCostFactoryTest(unittest.TestCase):
    def setUp(self):
        simple_sys = ampc.System(["x", "y"], ["u"])
        self.system = simple_sys
        self.Model = ARX
        self.model_cs = self.Model.get_configuration_space(simple_sys)
        self.model_cfg = self.model_cs.get_default_configuration()
        self.model = ampc.make_model(self.system, self.Model, self.model_cfg)

        # Initialize task
        Q = np.eye(2)
        R = np.eye(1)
        F = np.eye(2)
        cost = QuadCost(self.system, Q, R, F, goal=[-1,0])
        self.task = Task(self.system)
        self.task.set_cost(cost)
        self.task.set_ctrl_bound("u", -20.0, 20.0)

    def test_config_space(self):
        factory = QuadCostFactory()
        cs = factory.get_configuration_space(self.system, 
                self.task, self.Model)
        self.assertIsInstance(cs, CS.ConfigurationSpace)

        hyper_names = cs.get_hyperparameter_names()
        target_hyper_names = ["x_Q", "y_Q", "x_F", "y_F", "u_R"]
        self.assertEqual(set(hyper_names), set(target_hyper_names))

    def test_call_factory(self):
        factory = QuadCostFactory()
        cs = factory.get_configuration_space(self.system, 
                self.task, self.Model)
        cfg = cs.get_default_configuration()

        cost = factory(self.system, self.task, self.model, None, cfg)

        self.assertIsInstance(cost, QuadCost)
        Q, R, F = cost.get_cost_matrices()
        
        self.assertTrue((Q == np.eye(self.system.obs_dim)).all())
        self.assertTrue((F == np.eye(self.system.obs_dim)).all())
        self.assertTrue((R == np.eye(self.system.ctrl_dim)).all())
