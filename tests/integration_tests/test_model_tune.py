# Standard library includes
import unittest
import tempfile
import os

# External library includes
import numpy as np

# Internal library includes
import autompc as ampc
from autompc.tuning import ModelTuner, ModelTunerResult
from autompc.benchmarks import DoubleIntegratorBenchmark
from autompc.sysid import MLP

class ModelTuningIntegrationTest(unittest.TestCase):
    def setUp(self):
        # Init benchmark
        self.benchmark = DoubleIntegratorBenchmark()
        self.benchmark.task.set_num_steps(20)

        # Generate data
        self.trajs = self.benchmark.gen_trajs(seed=0, n_trajs=10, traj_len=10)

        # Set-Up AutoMPC output directory
        if os.getenv("AUTOMPC_OUTPUT_DIR"):
            self.autompc_dir = os.getenv("AUTOMPC_OUTPUT_DIR")
        else:
            self.autompc_dir = tempfile.mkdtemp()
        print(f"{self.autompc_dir=}")


    def test_model_tune(self):
        model = MLP(self.benchmark.system)

        tuner = ModelTuner(self.benchmark.system, self.trajs, model)
        tuned_model, tune_result = tuner.run(
            rng=np.random.default_rng(0),
            n_iters=5,
            output_dir=self.autompc_dir,
            eval_timeout=100
        )

        self.assertIsInstance(tuned_model, MLP)
        self.assertIsInstance(tune_result, ModelTunerResult)

if __name__ == "__main__":
    unittest.main()