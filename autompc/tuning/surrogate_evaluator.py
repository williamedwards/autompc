from collections.abc import Iterable, Callable
from multiprocessing.pool import Pool
import numpy as np
from .control_evaluator import ControlEvaluator, ConstantDistribution
from ..utils import simulate

class SurrogateEvaluator(ControlEvaluator):
    def __init__(self, system, tasks, trajs, surrogate, rng=None, 
            surrogate_mode="default", surrogate_tune_iters=100):
        super().__init__(system, tasks, trajs)
        self.surr_trajs = trajs
        self.surrogate = surrogate
        self.surrogate_mode = surrogate_mode
        if isinstance(tasks, Iterable):
            self.tasks = tasks[:]
        elif isinstance(tasks, Callable):
            raise NotImplementedError("SurrogateValuator does not support infinite task distribution")
        else: # Assume single task
            self.tasks = [tasks]

        if surrogate_mode != "pretrain":
            self._prepare_surrogate(trajs, rng, surrogate_tune_iters)

    def _prepare_surrogate(self, trajs, rng, surrogate_tune_iters):
        surrogate_tune_result = None
        if self.surrogate_mode == "default":
            self.surrogate = self.surrogate.clone()
            self.surrogate.train(trajs)
        elif self.surrogate_mode == "autotune":
            evaluator = self.surrogate_evaluator
            if evaluator is None:
                evaluator = HoldoutModelEvaluator(
                        holdout_prop = self.surrogate_tune_holdout,
                        metric = self.surrogate_tune_metric,
                        system=self.system,
                        trajs=trajs,
                        rng=rng)
            model_tuner = ModelTuner(self.system, evaluator) 
            model_tuner.add_model(self.surrogate)
            self.surrogate, self.surrogate_tune_result = model_tuner.run(rng, 
                    n_iters=surrogate_tune_iters) 
        elif self.surrogate_mode == "autoselect":
            evaluator = self.surrogate_evaluator
            if evaluator is None:
                evaluator = HoldoutModelEvaluator(
                        holdout_prop = self.surrogate_tune_holdout,
                        metric = self.surrogate_tune_metric,
                        system=self.system,
                        trajs=trajs,
                        rng=rng)
            model_tuner = ModelTuner(self.system, evaluator) 
            for model_class in autoselect_models:
                model_tuner.add_model(model_class(self.system))
            self.surrogate, sself.urrogate_tune_result = model_tuner.run(rng, 
                    n_iters=surrogate_tune_iters) 

    def _evaluate_for_task(self, controller, task):
        info = dict()
        print("Simulating Surrogate Trajectory: ")
        try:
            controller.set_ocp(task.get_ocp())
            controller.reset()
            if task.has_num_steps():
                surr_traj = simulate(controller, task.get_init_obs(),
                    task.term_cond, sim_model=self.surrogate, 
                    ctrl_bounds=task.get_ocp().get_ctrl_bounds(),
                    max_steps=task.get_num_steps())
            else:
                surr_traj = simulate(controller, task.get_init_obs(),
                    ctrl_bounds=task.get_ctrl_bounds(),
                    term_cond=task.term_cond, sim_model=surrogate)
            cost = task.get_ocp().get_cost()
            surr_cost = cost(surr_traj)
            print("Surrogate Cost: ", surr_cost)
            print("Surrogate Final State: ", surr_traj[-1].obs)
            info["surr_cost"] = surr_cost
            info["surr_traj"] = (surr_traj.obs.tolist(), surr_traj.ctrls.tolist())
        except np.linalg.LinAlgError:
            surr_cost = np.inf
            info["surr_cost"] = surr_cost
            info["surr_traj"] = None

        return surr_cost, info

    def __call__(self, controller):
        if len(self.tasks) == 1:
            surr_cost, info = self._evaluate_for_task(controller, self.tasks[0])
            distribution = ConstantDistribution(surr_cost)
            return distribution, info
        else:
            info = {"task_infos" : []}
            costs = []

            # Parallel Result Does Not Have Clear Debug Printout
            parallel = True
            results = []

            if(parallel):
                def setupTask(controllerIn, taskIn):
                    return self._evaluate_for_task(controllerIn, taskIn)

                pool = Pool()
                for i, task in enumerate(self.tasks):
                    print(f"Evaluating Task {i}")
                    results.append(pool.apply_async(func=setupTask, args=(controller,task)))
                    
                for res in results:
                    try:
                        output = res.get(timeout=1)
                    except TimeoutError:
                        print("TimeoutError In Surrogate Task Evaluation")
                    info["task_infos"].append(output[1])
                    costs.append(output[0])
                
                pool.close()
                pool.join()

            else:
                for i, task in enumerate(self.tasks):
                    print(f"Evaluating Task {i}")
                    task_cost, task_info = self._evaluate_for_task(controller, task)
                    info["task_infos"].append(task_info)
                    costs.append(task_cost)

                
            distribution = ConstantDistribution(np.mean(costs))
            return distribution, info
