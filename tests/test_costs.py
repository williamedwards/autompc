# Created by William Edwards (wre2@illinois.edu), 2021-01-24

# Standard library includes
import unittest

# Internal library includes
import autompc as ampc
from autompc.sysid import ARX
from autompc.costs import QuadCostFactory, QuadCost, GaussRegFactory
from autompc.tasks import Task

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

class GaussRegFactoryTest(unittest.TestCase):
    def setUp(self):
        double_int = ampc.System(["x", "y"], ["u"])
        self.system = double_int
        self.Model = ARX
        self.model_cs = self.Model.get_configuration_space(self.system)
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

        # Generate trajectories
        self.trajs = uniform_random_generate(double_int, self.task,
                lambda y,u: dt_doubleint_dynamics(y,u,dt=0.05),
                np.random.default_rng(42), init_min=[-1.0, -1.0],
                init_max=[1.0, 1.0], traj_len=20, n_trajs=20)

    def test_config_space(self):
        factory = GaussRegFactory()
        cs = factory.get_configuration_space(self.system, 
                self.task, self.Model)
        self.assertIsInstance(cs, CS.ConfigurationSpace)

        hyper_names = cs.get_hyperparameter_names()
        target_hyper_names = ["reg_weight"]
        self.assertEqual(set(hyper_names), set(target_hyper_names))

    def test_call_factory(self):
        factory = GaussRegFactory()
        cs = factory.get_configuration_space(self.system, 
                self.task, self.Model)
        cfg = cs.get_default_configuration()

        cost = factory(self.system, self.task, self.model, self.trajs, cfg)

        self.assertIsInstance(cost, QuadCost)
        Q, R, F = cost.get_cost_matrices()

        self.assertEqual(Q.shape, (self.system.obs_dim, self.system.obs_dim))
        self.assertTrue((F == np.zeros((self.system.obs_dim, self.system.obs_dim))).all())
        self.assertTrue((R == np.zeros((self.system.ctrl_dim, 
            self.system.ctrl_dim))).all())
