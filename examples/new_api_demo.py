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
from autompc.costs import ThresholdCost, QuadCost

ocp1 = ampc.OCP(system)
ocp1.set_cost(ThresholdCost(system, goal=np.zeros(4), threshold=0.2,
    obs_range=(0,4)))
ocp1.set_ctrl_bound("u", -10.0, 10.0)

ocp2 = ampc.OCP(system)
ocp2.set_cost(QuadCost(system, goal=np.zeros(4), Q=np.eye(4), R=np.eye(1), F=np.eye(4)))
ocp2.set_ctrl_bound("u", -10.0, 10.0)

task1 = ampc.Task(system)
task1.set_ocp(ocp1)
task1.set_init_obs(np.ones(4))
task1.set_num_steps(200)

task2 = ampc.Task(system)
task2.set_ocp(ocp2)
task2.set_init_obs(np.ones(4))
task2.set_num_steps(200)

task3 = ampc.Task(system)
task3.set_ocp(ocp2)
task3.set_init_obs(np.array([1.0,2.0,3.0,4.0]))
task3.set_num_steps(200)


## For models and controllers, I propose to remove the distinction
## between instance and factory.

from autompc.sysid import MLP

model = MLP(system)
model.train(trajs) # Model is trained under default configuration

model.clear() # Learned parameters are deleted and memory is released.

model_config_space = model.get_config_space()
model_config = model_config_space.get_default_configuration()
model_config["nonlintype"] = "relu"

model.set_config(model_config)
model.train(trajs)

## For optimizers...
from autompc.optim import IterativeLQR

optimizer = IterativeLQR(system)
optimizer.set_ocp(ocp2)
optimizer.set_model(model)
optimizer.reset() # Initialize any internal optimizer state

control = optimizer.run(np.zeros(4))

config = optimizer.get_default_config()
config["horizon"] = 15
optimizer.set_config(config)
optimizer.reset()

control2 = optimizer.run(np.zeros(4))

optimizer.set_ocp(ocp1)
optimizer.reset()


## For OCP transformation, I retain the factory distiniction.
from autompc.ocp import QuadCostFactory
ocp_factory = QuadCostFactory(system)

transformed_ocp1 = ocp_factory(ocp1)

config = ocp_factory.get_default_config()
config["x_Q"] = 10.0
ocp_factory.set_config(config)

transformed_ocp2 = ocp_factory(ocp2)

## Most of these details can be managed internally by the
## Controller class, which replaces Pipeline.

controller = ampc.Controller(system)

## The controller class internally manages multiple choices
## for model/optimizer/ocp_factory

from autompc.sysid import MLP, ARX
controller.add_model(MLP(system))
controller.add_model(ARX(system))

from autompc.optim import IterativeLQR, MPPI
controller.add_optimizer(IterativeLQR(system))
controller.add_optimizer(MPPI(system))

from autompc.ocp import QuadCostFactory
controller.add_ocp_factory(QuadCostFactory(system))

## The controller class creates the joint configuration
## space, including model/optimizer selection.
joint_config_space = controller.get_config_space()
joint_config = joint_config_space.get_default_configuration()
joint_config["optimizer"] = "MPPI"
controller.set_config(joint_config)

## We then provide the controller with the ocp and 
## and trajectories (if needed)

controller.set_trajs(trajs)
controller.set_ocp(ocp1)

## Calling build() initialize all internal components
## and trains the model
controller.build()

## We can than run the controller
control = controller.run(np.zeros(4))

## We can update the OCP without rebuilding
controller.set_ocp(ocp2)
controller.reset()

control = controller.run(np.zeros(4))

## Trying to run without building yields
## an error.

controller.clear()
try:
    control = controller.run(np.zeros(4))
except ampc.ControllerStateError:
    print("Can't run without building")

breakpoint()

## We run tuning by passing a prototype controller,
## a tuned controller is returned. The tuner clones
## the controller prototype internally as needed.
from autompc.tuning import ControlTuner
tuner = ControlTuner(surrogate=MLP(system), surrogate_split=0.5)
tuned_controller, tune_result = tuner.run(controller, [task1, task2, task3],
        trajs, n_iters=100, rng=np.random.default_rng(100))

