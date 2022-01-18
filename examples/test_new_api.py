import numpy as np
import sys
sys.path.insert(0, "..")
import autompc as ampc

# Initialize system and trajectories
from autompc.benchmarks import CartpoleSwingupBenchmark
benchmark = CartpoleSwingupBenchmark()

system = benchmark.system
trajs = benchmark.gen_trajs(seed=100, n_trajs=10, traj_len=50)

# Create task
from autompc.costs import ThresholdCost

ocp = ampc.OCP(system)
ocp.set_cost(ThresholdCost(system, goal=np.zeros(4), threshold=0.2,
    obs_range=(0,4)))
ocp.set_ctrl_bound("u", -10.0, 10.0)

task = ampc.Task(system)
task.set_ocp(ocp)
task.set_init_obs(np.ones(4))
task.set_num_steps(200)

# Create Controller
controller = ampc.Controller(system)

from autompc.sysid import MLP, ARX
controller.add_model(MLP(system))
controller.add_model(ARX(system))

from autompc.optim import IterativeLQR, MPPI
controller.add_optimizer(IterativeLQR(system))
controller.add_optimizer(MPPI(system))

from autompc.ocp import QuadCostFactory
controller.set_ocp_factory(QuadCostFactory(system))

cs = controller.get_config_space()

controller.set_trajs(trajs)
controller.set_ocp(ocp)
controller.build()

u = controller.run(np.ones(4))

breakpoint()
