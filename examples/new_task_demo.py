from autompc.tasks import StaticGoalTask
task = Task(system)

task.set_parameter("goal", )

QuadCost

TrajectoryTrackingCost

thershold_parameter = TaskParameter("threshold", default_value=0.2)

cost = ThresholdCost(system, threshold=threshold_parameter, goal=task.param_ref("goal"))
task.set_cost(cost)
task.set_num_steps(200)
task.set_init_obs(trajs[0][0].obs)
task.set_goal([1.0, 2.0])

task.get_parameter_names()
["threshold", "goal"]

quad_cost = cost_factory(task)

controller.update_parameter("goal", )

task.set_ctrl_bound("u", 0, 1)

task.set_parameter("goal", value)

perf_metric = task.get_cost()
assert perf_metric.goal == [1.0, 2.0]
perf_metric.update_task_parameter("goal", [3.0, 4.0])
assert perf_metric.goal == [3.0, 4.0]

quad_cost = cost_factory(cost_cfg, task)
assert quad_cost.goal == [1.0, 2.0]
quad_cost.update_task_parameter("goal", [5.0, 6.0])
assert quad_cost.goal == [5.0, 6.0]


controller_factory = optimizer_factory + cost_factory
cs = controller_factory.get_configuration_space()

controller = controller_factory(controller_cfg, model, task)
controller.set_task(task)

controller.set_task(task)

controller = controller_factory
controller.set_task(task)


from autompc.tasks import StaticGoalMultiTask
task = StaticGoalMultiTask(system)
task.set_default_parameter("goal", [1.0, 2.0])
task.set_default_parameter("init_obs", [0.0, 0.0])
task.add_subtask(goal=[1.0, 2.0])
task.add_subtask(goal=[3.0, 4.0])
task.add_subtask(goal=[5.0, 6.0], init_obs=[2.0, 5.0])



controller = Controller(system, cfg, trajs, ocp)

