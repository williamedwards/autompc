import sys
sys.path.append('..')
from autompc.benchmarks.pendulum import PendulumSwingupBenchmark
from autompc.controller import AutoSelectController

benchmark = PendulumSwingupBenchmark()
controller = AutoSelectController(benchmark.system)
controller.set_ocp(benchmark.task)
