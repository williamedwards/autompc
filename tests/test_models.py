# Created by William Edwards (wre2@illinois.edu), 2022-01-26

# Standard library includes
import unittest
import sys
from abc import ABC, abstractmethod

# Internal library inlcudes
sys.path.insert(0, "..")
import autompc as ampc
from autompc.sysid import MLP, ARX
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

    @abstractmethod
    def get_precomputed_prefix(self):
        raise NotImplementedError

    def get_inputs(self, num_inputs=100, seed=100):
        states = []
        controls = []
        ctrl_bounds = self.benchmark.task.get_ocp().get_ctrl_bounds()
        rng = np.random.default_rng(seed)
        for traj in self.trajs:
            for i in range(1, len(traj)):
                states.append(self.model.traj_to_state(traj[:i]))
                controls.append(rng.uniform(ctrl_bounds[0,0], ctrl_bounds[0,1]))
        states = np.array(states)
        controls = np.array(controls)
        if len(controls.shape) == 1:
            controls = controls.reshape((-1,1))
        selected_idxs = rng.choice(states.shape[0], num_inputs, replace=False)

        return states[selected_idxs, :], controls[selected_idxs, :]

    def get_precomputed(self, label):
        fn = f"{self.get_precomputed_prefix()}_{label}.txt"
        precomp = np.loadtxt(fn)
        return precomp

    def generate_precomputed(self):
        configs = self.get_configs_to_test()
        configs["default"] = self.model.get_default_config()
        input_states, input_controls = self.get_inputs()

        for label, config in configs.items():
            model = self.model.clone()
            model.set_config(config)
            model.train(self.trajs)
            preds = model.pred_batch(input_states, input_controls)
            fn = f"{self.get_precomputed_prefix()}_{label}.txt"
            np.savetxt(fn, preds)

    def test_model_train_predict(self):
        configs = self.get_configs_to_test()
        configs["default"] = self.model.get_default_config()
        input_states, input_controls = self.get_inputs()

        predss = []
        params = []
        for label, config in configs.items():
            model = self.model.clone()
            model.set_config(config)
            model.train(self.trajs)

            # Test Single Prediction
            preds1 = []
            for state, control in zip(input_states, input_controls):
                pred = model.pred(state, control)
                preds1.append(pred)
            preds1 = np.array(preds1)
            
            # Test Batch Prediction
            preds2 = model.pred_batch(input_states, input_controls)
            self.assertTrue(np.allclose(preds1, preds2))

            # Compare to pre-computed values
            precomp = self.get_precomputed(label)
            self.assertTrue(np.allclose(preds1, precomp))

            predss.append(preds1)
            params.append(model.get_parameters())

        # Check loading parameters
        for (label, config), preds, param in zip(configs.items(), predss, params):
            model = self.model.clone()
            model.set_config(config)
            model.set_parameters(param)

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

    def get_precomputed_prefix(self):
        return "precomputed/mlp"

class ARXTest(GenericModelTest, unittest.TestCase):
    def get_model(self, system):
        return ARX(system)

    def get_configs_to_test(self):
        return dict()

    def get_precomputed_prefix(self):
        return "precomputed/arx"


if __name__ == "__main__":
    if sys.argv[1] == "precompute":
        if sys.argv[2] == "mlp":
            test = MLPTest()
        elif sys.argv[2] == "arx":
            test = ARXTest()
        else:
            raise ValueError("Unknown model")
        test.setUp()
        test.generate_precomputed()
    else:
        raise ValueError("Unknown command")
