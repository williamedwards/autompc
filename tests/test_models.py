# Created by William Edwards (wre2@illinois.edu), 2022-01-26

# Standard library includes
import unittest
import sys
from abc import ABC, abstractmethod

# Internal library inlcudes
sys.path.insert(0, "..")
import autompc as ampc
from autompc.sysid import MLP, ARX, SINDy, ApproximateGPModel, Koopman
from autompc.benchmarks import DoubleIntegratorBenchmark


# External library includes
import numpy as np

class GenericModelTest(ABC):
    def setUp(self):
        self.benchmark = DoubleIntegratorBenchmark()
        self.trajs = self.benchmark.gen_trajs(seed=100, n_trajs=20, traj_len=20)
        self.model = self.get_model(self.benchmark.system)

    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_configs_to_test(self):
        """Return non-default configs to test, default is
        always included.  Returned values should be dictionary, mapping
        label name to config."""
        raise NotImplementedError

    def get_inputs(self, model, num_inputs=100, seed=100):
        states = []
        controls = []
        ctrl_bounds = self.benchmark.task.get_ocp().get_ctrl_bounds()
        rng = np.random.default_rng(seed)
        for traj in self.trajs:
            for i in range(1, len(traj)):
                states.append(model.traj_to_state(traj[:i]))
                controls.append(rng.uniform(ctrl_bounds[0,0], ctrl_bounds[0,1]))
        states = np.array(states)
        controls = np.array(controls)
        if len(controls.shape) == 1:
            controls = controls.reshape((-1,1))
        selected_idxs = rng.choice(states.shape[0], num_inputs, replace=False)

        return states[selected_idxs, :], controls[selected_idxs, :]

    def test_model_train_predict(self):
        configs = self.get_configs_to_test()
        configs["default"] = self.model.get_default_config()

        predss = []
        params = []
        for label, config in configs.items():
            model = self.model.clone()
            model.set_config(config)
            model.train(self.trajs)
            input_states, input_controls = self.get_inputs(model)

            self.assertEqual(input_states.shape[1], model.state_dim)

            # Test Single Prediction
            preds1 = []
            for state, control in zip(input_states, input_controls):
                pred = model.pred(state, control)
                self.assertEqual(pred.ndim, 1)
                self.assertEqual(pred.size, model.state_dim)
                preds1.append(pred)
            preds1 = np.array(preds1)
            
            # Test Batch Prediction
            preds2 = model.pred_batch(input_states, input_controls)
            self.assertEqual(preds2.ndim, 2)
            self.assertEqual(preds2.shape[0], input_states.shape[0])
            self.assertEqual(preds2.shape[1], model.state_dim)
            self.assertTrue(np.allclose(preds1, preds2))

            predss.append(preds1)
            params.append(model.get_parameters())

        # Check loading parameters
        for (label, config), preds, param in zip(configs.items(), predss, params):
            model = self.model.clone()
            model.set_config(config)
            model.set_parameters(param)
            input_states, input_controls = self.get_inputs(model)

            preds2 = model.pred_batch(input_states, input_controls)
            self.assertTrue(np.allclose(preds, preds2))


class MLPTest(GenericModelTest, unittest.TestCase):
    def get_model(self, system):
        return MLP(system)

    def get_configs_to_test(self):
        relu_config = self.model.get_default_config()
        relu_config["nonlintype"] = "relu"
        tanh_config = self.model.get_default_config()
        tanh_config["nonlintype"] = "tanh"
        sigmoid_config = self.model.get_default_config()
        sigmoid_config["nonlintype"] = "sigmoid"
        return {"relu" : relu_config,
                "tanh" : tanh_config,
                "sigmoid" : sigmoid_config}

class ARXTest(GenericModelTest, unittest.TestCase):
    def get_model(self, system):
        return ARX(system)

    def get_configs_to_test(self):
        return dict()

class SINDyTest(GenericModelTest, unittest.TestCase):
    def get_model(self, system):
        return SINDy(system, allow_cross_terms=True)

    def get_configs_to_test(self):
        poly_config = self.model.get_default_config()
        poly_config["poly_basis"] = "true"
        poly_config["poly_degree"] = 3
        poly_config["poly_cross_terms"] = "true"
        
        trig_config = self.model.get_default_config()
        trig_config["trig_basis"] = "true"
        trig_config["trig_freq"] = 3
        trig_config["trig_interaction"] = "true"

        trig_and_poly_config = self.model.get_default_config()
        trig_and_poly_config["poly_basis"] = "true"
        trig_and_poly_config["poly_degree"] = 2
        trig_and_poly_config["trig_basis"] = "true"
        trig_and_poly_config["trig_freq"] = 2

        return {"poly" : poly_config,
                "trig" : trig_config,
                "trig_and_poly" : trig_and_poly_config
                }


class ApproximateGPTest(GenericModelTest, unittest.TestCase):
    def get_model(self, system):
        return ApproximateGPModel(system)

    def get_configs_to_test(self):
        return dict()

class KoopmanTest(GenericModelTest, unittest.TestCase):
    def get_model(self, system):
        return Koopman(system, allow_cross_terms=True)

    def get_configs_to_test(self):
        poly_config = self.model.get_default_config()
        poly_config["poly_basis"] = "true"
        poly_config["poly_degree"] = 3
        poly_config["poly_cross_terms"] = "true"
        
        trig_config = self.model.get_default_config()
        trig_config["trig_basis"] = "true"
        trig_config["trig_freq"] = 3
        trig_config["trig_interaction"] = "true"

        trig_and_poly_config = self.model.get_default_config()
        trig_and_poly_config["poly_basis"] = "true"
        trig_and_poly_config["poly_degree"] = 2
        trig_and_poly_config["trig_basis"] = "true"
        trig_and_poly_config["trig_freq"] = 2

        lasso_config = self.model.get_default_config()
        lasso_config["method"] = "lasso"
        lasso_config["poly_basis"] = "true"
        lasso_config["poly_degree"] = 2
        lasso_config["trig_basis"] = "true"
        lasso_config["trig_freq"] = 2

        stable_config = self.model.get_default_config()
        stable_config["method"] = "stable"
        stable_config["poly_basis"] = "true"
        stable_config["poly_degree"] = 2
        stable_config["trig_basis"] = "true"
        stable_config["trig_freq"] = 2

        return {"poly" : poly_config,
                "trig" : trig_config,
                "trig_and_poly" : trig_and_poly_config,
                "stable" : stable_config,
                "lasso" : lasso_config}