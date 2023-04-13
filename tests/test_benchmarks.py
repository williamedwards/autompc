# Created by William Edwards (wre2@illinois.edu), 2022-12-15

# Standard libary includes
import unittest
from abc import ABC, abstractmethod

# External library includes
import numpy as np

# Internal library includes
from autompc.benchmarks import CartpoleSwingupBenchmark, ControlBenchmark 
from autompc.benchmarks.gym_mujoco import GymMujocoBenchmark
from autompc.benchmarks.gym_reacher import GymReacherBenchmark
from autompc.trajectory import Trajectory
from autompc.system import System
from autompc.task import Task
from autompc.costs.cost import Cost

class GenericBenchmarkTest(ABC):
    def setUp(self):
        self.benchmark = self.get_benchmark()

    @abstractmethod
    def get_benchmark(self) -> ControlBenchmark:
        raise NotImplementedError 

    def test_gen_trajs(self):
        trajs = self.benchmark.gen_trajs(seed=0, n_trajs=10, traj_len=15)
        self.assertEqual(len(trajs), 10)
        for traj in trajs:
            self.assertIsInstance(traj, Trajectory)
            self.assertEqual(len(traj), 15)

    def test_properties(self):
        self.assertIsInstance(self.benchmark.system, System)
        self.assertIsInstance(self.benchmark.task, Task)
        self.assertIsInstance(self.benchmark.task.get_cost(), Cost)

    def test_cost(self):
        cost = self.benchmark.task.get_cost()
        traj = self.benchmark.gen_trajs(seed=0, n_trajs=1, traj_len=100)[0]
        cost_value = cost(traj)
        
        self.assertTrue(np.isscalar(cost_value))
        self.assertTrue(np.isfinite(cost_value))

class CartpoleSwingupBenchmarkTest(GenericBenchmarkTest, unittest.TestCase):
    def get_benchmark(self):
        return CartpoleSwingupBenchmark()

class HalfcheetahBenchmark(GenericBenchmarkTest, unittest.TestCase):
    def get_benchmark(self):
        return GymMujocoBenchmark(name="HalfCheetah-v2")

class HopperBenchmark(GenericBenchmarkTest, unittest.TestCase):
    def get_benchmark(self):
        return GymMujocoBenchmark(name="Hopper-v2")

class AntBenchmark(GenericBenchmarkTest, unittest.TestCase):
    def get_benchmark(self):
        return GymMujocoBenchmark(name="Ant-v2")

class HumanoidBenchmark(GenericBenchmarkTest, unittest.TestCase):
    def get_benchmark(self):
        return GymMujocoBenchmark(name="Humanoid-v2")

class ReacherBenchmark(GenericBenchmarkTest, unittest.TestCase):
    def get_benchmark(self):
        return GymReacherBenchmark()