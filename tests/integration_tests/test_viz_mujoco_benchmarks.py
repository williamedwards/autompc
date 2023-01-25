 # Created by William Edwards (wre2@illinois.edu)

# Standard library includes
import unittest
from abc import ABC, abstractmethod

# External library includes
import numpy as np

# Internal library includes
from autompc.benchmarks.control_benchmark import ControlBenchmark
from autompc.benchmarks.gym_mujoco import GymMujocoBenchmark


class GenericMujocoBenchmarkVisualizationTest(ABC):
    def setUp(self):
        self.benchmark = self.get_benchmark()
        self.traj = self.benchmark.gen_trajs(seed=0, n_trajs=1, traj_len=200)[0]

    @abstractmethod
    def get_benchmark(self) -> ControlBenchmark:
        raise NotImplementedError

    def test_visualization(self):
        print(f"Visualizing Environment: {self.benchmark.env_name}")

        # First, check trajectory cost
        traj_cost = self.benchmark.task.get_cost()(self.traj)
        print(f"Trajectory Cost: {traj_cost}")

        # Next, visualize the trajectory
        out_path = f"gym_video_out/{self.benchmark.env_name}"
        self.benchmark.visualize(self.traj, repeat=1, file_path=out_path)

class HalfcheetahVisualizationTest(GenericMujocoBenchmarkVisualizationTest, unittest.TestCase):
    def get_benchmark(self):
        return GymMujocoBenchmark(name="HalfCheetah-v2")

class HopperVisualizationTest(GenericMujocoBenchmarkVisualizationTest, unittest.TestCase):
    def get_benchmark(self):
        return GymMujocoBenchmark(name="Hopper-v2")

class HumanoidVisualizationTest(GenericMujocoBenchmarkVisualizationTest, unittest.TestCase):
    def get_benchmark(self):
        return GymMujocoBenchmark(name="Humanoid-v2")

class ReacherVisualizationTest(GenericMujocoBenchmarkVisualizationTest, unittest.TestCase):
    def get_benchmark(self):
        return GymMujocoBenchmark(name="Reacher-v2")

class IDPVisualizationTest(GenericMujocoBenchmarkVisualizationTest, unittest.TestCase):
    def get_benchmark(self):
        return GymMujocoBenchmark(name="InvertedDoublePendulum-v2")