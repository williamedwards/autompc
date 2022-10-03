# Standard library includes
import unittest
import tempfile

# External library includes
import numpy as np

# Internal library includes
import autompc as ampc
from autompc.tuning import ControlTuner, ControlTunerResult
from autompc.benchmarks import DoubleIntegratorBenchmark
from autompc.sysid import MLP
from autompc import Controller, AutoSelectController
from autompc.sysid import MLP
from autompc.optim import IterativeLQR
from autompc.ocp import QuadCostTransformer


class ControlTuningIntegrationTest(unittest.TestCase):
    def setUp(self):
        # Init benchmark
        self.benchmark = DoubleIntegratorBenchmark()
        self.benchmark.task.set_num_steps(20)

        # Generate data
        self.trajs = self.benchmark.gen_trajs(seed=0, n_trajs=10, traj_len=10)

        # Set-Up Temporary Directories
        self.autompc_dir = tempfile.mkdtemp()
        print(f"{self.autompc_dir=}")

        # Surrogate
        self.surrogate = MLP(self.benchmark.system)
        self.surrogate.freeze_hyperparameters()

    def test_control_tune(self):
        # controller = AutoSelectController(self.benchmark.system)
        controller = Controller(self.benchmark.system)
        controller.add_optimizer(IterativeLQR(self.benchmark.system))
        controller.add_model(MLP(self.benchmark.system, n_train_iters=10))
        controller.add_ocp_transformer(QuadCostTransformer(self.benchmark.system))

        tuner = ControlTuner(surrogate=self.surrogate, surrogate_split=0.5)
        tuned_controller, tune_result = tuner.run(
            controller, 
            self.benchmark.task,
            self.trajs,
            n_iters=5,
            rng = np.random.default_rng(0),
            truedyn=self.benchmark.dynamics,
            output_dir=self.autompc_dir
        )

        self.assertIsInstance(tuned_controller, Controller)
        self.assertIsInstance(tune_result, ControlTunerResult)