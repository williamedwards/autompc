import sys
sys.path.insert(0, "..")

import autompc as ampc
from autompc.sysid import MLPFactory

from autompc.benchmarks import CartpoleSwingupBenchmark

benchmark = CartpoleSwingupBenchmark()
system  = benchmark.system
trajs = benchmark.gen_trajs(n_trajs=10, traj_len=10, seed=100)

factory = MLPFactory(system)
cs = factory.get_configuration_space()
cfg = cs.get_default_configuration()

breakpoint()

model = factory(cfg, trajs)
partial_factory = factory(cfg)
model2 = partial_factory(trajs)


breakpoint()