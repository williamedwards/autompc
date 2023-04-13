# Created by William Edwards (wre2@illinois.edu), 2022-01-26

# Standard library includes
import unittest
import sys
from abc import ABC, abstractmethod

# Internal library inlcudes
sys.path.insert(0, "..")
import autompc as ampc
from autompc.sysid import MLP, ARX
from autompc.optim import (IterativeLQR, MPPI, LQR, DirectTranscription, 
                           ZeroOptimizer, RoundedOptimizer)
from autompc.ocp import OCP
from autompc.costs import QuadCost
from autompc.benchmarks import DoubleIntegratorBenchmark


# External library includes
import numpy as np

class GenericOptimTest(ABC):
    def setUp(self):
        self.benchmark = DoubleIntegratorBenchmark()
        self.trajs = self.benchmark.gen_trajs(seed=100, n_trajs=20, traj_len=20)
        self.model = self.get_model(self.benchmark.system)
        self.ocp = self.get_ocp(self.benchmark.system)
        self.optim = self.get_optim(self.benchmark.system)

    def get_model(self, system):
        model = MLP(system, n_train_iters=5)
        model.train(self.trajs)
        return model

    def get_ocp(self, system):
        ocp = OCP(system)
        ocp.set_cost(QuadCost(system, Q=np.eye(2), F=np.eye(2), R=np.eye(1)))
        ocp.set_ctrl_bound("u", -10.0, 10.0)
        return ocp

    @abstractmethod
    def get_configs_to_test(self):
        """Return non-default configs to test, default is
        always included.  Returned values should be dictionary, mapping
        label name to config."""
        raise NotImplementedError

    def get_inputs(self, num_inputs=10, seed=100):
        states = []
        rng = np.random.default_rng(seed)
        for traj in self.trajs:
            for i in range(1, len(traj)):
                states.append(self.model.traj_to_state(traj[:i]))
        states = np.array(states)
        selected_idxs = rng.choice(states.shape[0], num_inputs, replace=False)

        return states[selected_idxs, :]

    def test_optim_run(self):
        configs = self.get_configs_to_test()
        configs["default"] = self.optim.get_default_config()
        inputs = self.get_inputs()

        for label, config in configs.items():
            optim = self.optim.clone()
            self.assertTrue(optim.is_compatible(self.model, self.ocp))
            optim.set_config(config)
            optim.set_ocp(self.ocp)
            optim.set_model(self.model)
            controls = np.zeros((inputs.shape[0], self.benchmark.system.ctrl_dim))
            optim.reset()
            for i, inp in enumerate(inputs):
                if i == 5:
                    optim_state = optim.get_state()
                controls[i,:] = optim.step(inp)

            optim.reset()
            controls_2 = np.zeros((inputs.shape[0], self.benchmark.system.ctrl_dim))
            for i, inp in enumerate(inputs):
                controls_2[i,:] = optim.step(inp)
            self.assertTrue(np.allclose(controls, controls_2))

            controls_3 = np.zeros((inputs.shape[0]-5, self.benchmark.system.ctrl_dim))
            optim.set_state(optim_state)
            for i, inp in enumerate(inputs[5:]):
                controls_3[i,:] = optim.step(inp)

            self.assertTrue(np.allclose(controls[5:,:], controls_3))

class IterativeLQRTest(GenericOptimTest, unittest.TestCase):
    def get_optim(self, system):
        return IterativeLQR(system)

    def get_configs_to_test(self):
        return dict()

class RoundedIterativeLQRTest(GenericOptimTest, unittest.TestCase):
    def get_optim(self, system):
        return RoundedOptimizer(system, IterativeLQR(system))

    def get_configs_to_test(self):
        return dict()

class MPPITest(GenericOptimTest, unittest.TestCase):
    def get_optim(self, system):
        return MPPI(system)

    def get_configs_to_test(self):
        return dict()

class DirectTranscriptionTest(GenericOptimTest, unittest.TestCase):
    def get_optim(self, system):
        return DirectTranscription(system)

    def get_configs_to_test(self):
        return dict()

class ZeroTest(GenericOptimTest, unittest.TestCase):
    def get_optim(self, system):
        return ZeroOptimizer(system)

    def get_configs_to_test(self):
        return dict()

class LQRTest(GenericOptimTest, unittest.TestCase):
    def get_optim(self, system):
        return LQR(system)

    def get_model(self, system):
        model = ARX(system)
        config = model.get_default_config()
        config["history"] = 1
        model.set_config(config)
        model.train(self.trajs)
        return model

    def get_configs_to_test(self):
        config = self.optim.get_default_config()
        config["finite_horizon"] = "false"
        return {"infinite" : config}