
# Standard library includes
from collections import namedtuple
import pickle
import queue
import multiprocessing
import contextlib
import datetime
import time
import os, sys, glob, shutil
from typing import Optional, List

# Internal project include
from ..task import Task
from ..sysid.model import Model
from ..sysid.autoselect import AutoSelectModel
from ..controller import Controller
from ..trajectory import Trajectory
from ..dynamics import Dynamics
from .model_tuner import ModelTuner
from .model_evaluator import ModelEvaluator
from .control_evaluator import ControlEvaluator, StandardEvaluator, ControlEvaluationTrial, trial_to_json
from .control_performance_metric import ControlPerformanceMetric,ConfidenceBoundPerformanceMetric
from .bootstrap_evaluator import BootstrapSurrogateEvaluator
from .parallel_evaluator import ParallelEvaluator

# External library includes
import numpy as np
import pynisher
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger


ControlTunerResult = namedtuple("ControlTunerResult", ["inc_cfg", "cfgs", 
    "inc_cfgs", "costs", "inc_costs", "truedyn_costs", "inc_truedyn_costs", 
    "truedyn_infos", "surr_infos", 
    "surr_tune_result"]) # TODO update docstring
"""
PipelineTuneResult contains information about a tuning process.

.. py:attribute:: inc_cfg
    
    The final tuned configuration.

.. py:attribute:: cfgs

    List of configurations. The configuration evaluated at each
    tuning iteration.

.. py:attribute:: inc_cfgs

    The incumbent (best found so far) configuration at each
    tuning iteration.

.. py:attribute:: costs

    The cost observed at each iteration of
    tuning iteration.

.. py:attribute:: inc_costs

    The incumbent score at each tuning iteration.

.. py:attribute:: truedyn_costs

    The true dynamics cost observed at each iteration of
    tuning iteration. None if true dynamics was not provided.

.. py:attribute:: inc_truedyn_costs

    The incumbent true dynamics cost observed at each iteration of
    tuning iteration. None if true dynamics was not provided.

.. py:attribute:: surr_trajs

    The trajectory simulated at each tuning iteration with respect to the
    surrogate model.

.. py:attribute:: truedyn_trajs

    The trajectory simulated at each tuning iteration with respect to the
    true dynamics. None if true dynamics are not provided.

.. py:attribute:: surr_tune_result

    The ModelTuneResult from tuning the surrogate model, for modes "autotune"
    and "autoselect".  None for other tuning modes.

"""

#autoselect_factories = [MLPFactory, SINDyFactory, ApproximateGPModelFactory,
#        ARXFactory, KoopmanFactory]


class ControlTuner:
    """
    This class tunes controllers, searching over the hyperparameters for the
    model, optimizer, and ocp_factory in order to optimize a performance metric.

    There are three major tuning modes:

    - End-to-end: tunes end-to-end performance of model, cost, and optimizer.
      Specify tuning_mode='default' or 'endtoend'.
    - Two-stage: tunes the model for accuracy first, then tunes the cost
      and optimizer assuming the model is fixed.  This is usually faster
      because the hyperparameter space for each stage is smaller.
      Specify tuning_mode='twostage'.
    - Fixed-model: uses a given model and tunes the cost and optimizer only.
      Use this only if you know the dynamics model.  Specify surrogate=X where
      X is a Dynamics instance or a trained Model instance.

    In all but fixed-model mode, cross-simulation is performed.  In cross-
    simulation, one or more surrogate models are learned on a holdout set,
    and the controller's models are learned from the remainder.  The surrogate
    model is auto-tuned on the holdout set.  The controller's
    performance is tuned on the surrogate model(s).

    The surrogate model is tuned on the holdout set according to some accuracy-
    related performance metric.  By default, a CrossValidationModelEvaluator with
    surrogate_tune_folds folds is used. 
    
    - To use holdout evaluation, you can use surrogate_tune_folds = 1 and specify
      surrogate_tune_holdout.
    - To use a custom evaluator, you can specify surrogate_evaluator=X. 
    - If you want to fix the surrogate model class, pass in surrogate=X where X
      is a Model instance (untrained). 
    - If you want to fix the class and its hyperparameters, pass in surrogate=X
      and call `freeze_hyperparameters()` on X.
    
    In other words, the arguments `surrogate_` map directly to the arguments of
    ModelTuner.

    The cross-simulation method by default runs a HoldoutSurrogateEvaluator
    with a holdout fraction of control_tune_holdout. A BootstrapSurrogateEvaluator
    will run many surrogates, and can be activated using control_tune_boostraps > 1.
    To completely override this behavior, specify control_evaluator=X. 
    
    The performance metric is encapsulated by a ControlPerformanceMetric subclass. 
    The default tuner will minimize the performance distribution across surrogates
    and tasks at quantile `evaluation_quantile`.

    To freeze some aspects of the Controller, pass in controller=X where X is a
    configured Controller with the desired `set_optimizer_hyper_values` and
    `set_ocp_transformer_hyper_values` items fixed.
    """
    def __init__(self, tuning_mode="default", surrogate : Optional[Model] = None,
            surrogate_split=0.5,
            surrogate_tune_holdout=0.25, surrogate_tune_folds=3, surrogate_tune_metric="rmse", surrogate_tune_horizon=1, surrogate_tune_quantile=None,
            surrogate_evaluator : Optional[ModelEvaluator]=None,
            control_tune_bootstraps=1, max_trials_per_evaluation=None, control_evaluator : Optional[ControlEvaluator]=None,
            performance_quantile=0.5, performance_eval_time_weight=0.0, performance_infeasible_cost=100.0,
            performance_metric : Optional[ControlPerformanceMetric]=None): 
        """
        Parameters
        ----------
        tuning_mode : string
            Mode for selecting tuning mode
            "default" or "endtoend" - whole controller will be tuned at once
            "twostage" - controller model will be tuned first, then the cost
                and optimizer
        surrogate : Model
            Surrogate model. If None, a model will be auto-selected in the first step.
            If tunable, this will be tuned in the first step.  If learnable
            and has not been learned, then this will be passed to a holdout or bootstrap
            control evaluator.
        surrogate_split : float
            The fraction of examples used for training the surrogate vs
            the controller's sysID model. If no surrogate tuning is
            performed, then all the data will be used for control tuning.
        surrogate_tune_holdout : float
            Passed into ModelTuner as the eval_holdout argument.
        surrogate_tune_folds : int
            Passed into ModelTuner as the eval_folds argument.
        surrogate_tune_metric : int
            Passed into ModelTuner as the eval_metric argument.
        surrogate_tune_horizon : int
            Passed into ModelTuner as the horizon argument.
        surrogate_tune_quantile : float or None
            Passed into ModelTuner as the eval_quantile argument.
        surrogate_evaluator : int
            Passed into ModelTuner as the evaluator argument.  If given,
            ignores the prior settings.
        control_tune_bootstraps : int
            If > 1 and control_evaluator = None, will use a
            BootstrapControlEvaluator. Otherwise will use a
            SurrogateControlEvaluator.
        max_trials_per_evaluation : int
            If given and control_evaluator = None, at most this # of tasks and 
            surrogates will be evaluated.
        control_evaluator : ControlEvaluator
            Overrides the method by which controllers will be evaluated and ignores
            the prior settings.
        performance_quantile : float
            Optimizes this quantile of the distribution of task / surrogate
            performance.
        performance_eval_time_weight : float
            Penalizes the evaluation time of the controller (in seconds) by
            this weight in the overall performance metric.
        performance_infeasible_cost : float
            Penalizes infeasible states / controls by this much in the overall
            performance metric
        performance_metric : ControlPerformanceMetric
            Overrides the default ConfidenceBoundPerformanceMetric and ignores the
            prior settings.
        """
        self.tuning_mode = tuning_mode
        self.surrogate = surrogate
        self.surrogate_tune_holdout = surrogate_tune_holdout
        self.surrogate_tune_folds = surrogate_tune_folds
        self.surrogate_tune_metric = surrogate_tune_metric
        self.surrogate_tune_horizon = surrogate_tune_horizon
        self.surrogate_tune_quantile = surrogate_tune_quantile
        self.surrogate_evaluator = surrogate_evaluator
        self.surrogate_split = surrogate_split
        self.control_tune_bootstraps = control_tune_bootstraps
        self.max_trials_per_evaluation = max_trials_per_evaluation
        self.control_evaluator = control_evaluator
        if performance_metric is None:
            performance_metric = ControlPerformanceMetric()
            #performance_metric = ConfidenceBoundPerformanceMetric(quantile=performance_quantile,eval_time_weight=performance_eval_time_weight,infeasible_cost=performance_infeasible_cost) # TODO: Fix this
        self.performance_metric = performance_metric

    def _get_tuning_data(self, controller : Controller, task : List[Task], trajs : List[Trajectory],
                            truedyn : Optional[Dynamics], rng, surrogate_tune_iters : int):
        surrogate = self.surrogate
        if surrogate is None:
            surrogate = AutoSelectModel(controller.system)
        control_evaluator = self.control_evaluator
        surrogate_split = self.surrogate_split
        if surrogate.is_trained:
            # No need for surrogate training or tuning
            surrogate_split = 0.0
        
        surr_size = int(surrogate_split * len(trajs))
        shuffled_trajs = trajs[:]
        rng.shuffle(shuffled_trajs)
        surr_trajs = shuffled_trajs[:surr_size]
        sysid_trajs = shuffled_trajs[surr_size:]

        tuning_data = dict()
        tuning_data["controller"] = controller
        tuning_data["performance_metric"] = self.performance_metric
        tuning_data["sysid_trajs"] = sysid_trajs

        if surrogate.is_trained:
            print("------------------------------------------------------------------")
            print("Skipping surrogate tuning, surrogate is a trained model")
            print("------------------------------------------------------------------")
            if control_evaluator is None:
                control_evaluator = ParallelEvaluator(StandardEvaluator(controller.system, task, surrogate, 'surr_'), dynamics=surrogate, max_jobs=len(task))
            else:
                assert not isinstance(control_evaluator,BootstrapSurrogateEvaluator),'Need an evaluator that does not train'
        else:
            if surrogate.is_tunable():
                # Run surrogate tuning
                print("------------------------------------------------------------------")
                print("Beginning surrogate tuning with model class",surrogate.name)
                print("------------------------------------------------------------------")
                tuner = ModelTuner(controller.system, surr_trajs, surrogate, eval_holdout=self.surrogate_tune_holdout, eval_folds=self.surrogate_tune_folds,
                    eval_metric=self.surrogate_tune_metric,eval_horizon=self.surrogate_tune_horizon,eval_quantile=self.surrogate_tune_quantile,evaluator=self.surrogate_evaluator)
                model, tune_result = tuner.run(rng, surrogate_tune_iters, retrain_full=False)
                print("------------------------------------------------------------------")
                print("Auto-tuned surrogate model class:")
                print(tune_result.inc_cfg)
                print("Cost",tune_result.inc_costs[-1])
                print("------------------------------------------------------------------")
                tuning_data['surr_tune_result'] = tune_result
                surrogate = model
            else:
                # No need for surrogate tuning, but would need training
                pass
        
            if control_evaluator is None:
                if self.control_tune_bootstraps > 1:
                    control_evaluator = BootstrapSurrogateEvaluator(controller.system, task, surrogate, surr_trajs, self.control_tune_bootstraps, rng=rng)
                else:
                    surrogate.train(surr_trajs)
                    control_evaluator = StandardEvaluator(controller.system, task, surrogate, prefix='surr_')
            else:
                if isinstance(control_evaluator,StandardEvaluator):
                    surrogate.train(surr_trajs)
                    control_evaluator.dynamics = surrogate
                else:
                    raise NotImplementedError("Not sure what to do when control_evaluator != None and surrogate is tuned?")

        if truedyn is not None:
            truedyn_evaluator = StandardEvaluator(controller.system, task, dynamics = truedyn, prefix='truedyn_')
        else:
            truedyn_evaluator = None
        
        if self.tuning_mode == 'twostage':
            #tune the model if not already tuned by surrogate tuning
            print("------------------------------------------------------------------")
            print("Beginning two-stage tuning")
            print("------------------------------------------------------------------")
            if 'surr_tune_result' in tuning_data.keys() and isinstance(surrogate,AutoSelectModel):
                print("Reusing surrogate model tuning for SysID model to save time")
                model = surrogate
            else:
                print("Tuning MPC model")
                tuner = ModelTuner(controller.system, trajs, surrogate, eval_holdout=self.surrogate_tune_holdout, eval_folds=self.surrogate_tune_folds,
                    eval_metric=self.surrogate_tune_metric,eval_horizon=self.surrogate_tune_horizon,eval_quantile=self.surrogate_tune_quantile,evaluator=self.surrogate_evaluator)
                model, tune_result = tuner.run(rng, surrogate_tune_iters, retrain_full=False)
                print("------------------------------------------------------------------")
                print("Auto-tuned SysID model class:")
                print(tune_result.inc_cfg)
                print("Cost",tune_result.inc_costs[-1])
                print("------------------------------------------------------------------")
            controller.fix_option('model',model.name)
            controller.set_model_hyper_values(model.name,**tune_result.inc_cfg)
            controller.model.freeze_hyperparameters()
            
            print()
        
        tuning_data["control_evaluator"] = control_evaluator
        tuning_data["truedyn_evaluator"] = truedyn_evaluator

        return tuning_data

    def _get_restore_run_dir(self, restore_dir):
        run_dirs = glob.glob(os.path.join(restore_dir, "run_*"))
        for run_dir in reversed(sorted(run_dirs)):
            if os.path.exists(os.path.join(run_dir, "smac", "run_1", "runhistory.json")):
                return run_dir
        raise FileNotFoundError("No valid restore files found")

    def _copy_restore_data(self, restore_run_dir, new_run_dir):
        # Copy tuning data
        old_tuning_data = os.path.join(restore_run_dir, "tuning_data.pkl")
        new_tuning_data = os.path.join(new_run_dir, "tuning_data.pkl")
        shutil.copy(old_tuning_data, new_tuning_data)
        # Copy log
        old_log = os.path.join(restore_run_dir, "log.txt")
        new_log = os.path.join(new_run_dir, "log.txt")
        shutil.copy(old_log, new_log)
        # Copy smac trajectory information
        old_traj = os.path.join(restore_run_dir, "smac", "run_1", "traj_aclib2.json")
        new_traj = os.path.join(new_run_dir, "smac", "run_1", "traj_aclib2.json")
        shutil.copy(old_traj, new_traj)

    def _load_smac_restore_data(self, restore_run_dir, scenario):
        # Load runhistory
        rh_path = os.path.join(restore_run_dir, "smac", "run_1", "runhistory.json")
        runhistory = RunHistory()
        runhistory.load_json(rh_path, scenario.cs)
        # Load stats
        stats_path = os.path.join(restore_run_dir, "smac", "run_1", "stats.json")
        stats = Stats(scenario)
        stats.load(stats_path)
        # Load trajectory
        traj_path = os.path.join(restore_run_dir, "smac", "run_1", "traj_aclib2.json")
        trajectory = TrajLogger.read_traj_aclib_format(fn=traj_path, cs=scenario.cs)
        incumbent = trajectory[-1]["incumbent"]

        return runhistory, stats, incumbent

    def _get_tune_result(self, tuning_data, runhistory):
        cfgs, inc_cfgs, costs, inc_costs, truedyn_costs, inc_truedyn_costs, \
            surr_infos, truedyn_infos =  [], [], [], [], [], [], [], []
        inc_cost = float("inf")

        for key, val in runhistory.data.items():
            #parse additional data from additional_info dict
            cfg = runhistory.ids_config[key.config_id]
            if not val.additional_info:
                continue
            if val.cost < inc_cost:
                inc_cost = val.cost
                if "truedyn_cost" in val.additional_info:
                    inc_truedyn_cost = val.additional_info["truedyn_cost"]
                inc_cfg = cfg
            inc_costs.append(inc_cost)
            inc_cfgs.append(inc_cfg)
            cfgs.append(cfg)
            costs.append(val.cost)
            surr_infos.append(val.additional_info["surr_info"])
            if "truedyn_cost" in val.additional_info:
                inc_truedyn_costs.append(inc_truedyn_cost)
                truedyn_costs.append(val.additional_info["truedyn_cost"])
                truedyn_infos.append(val.additional_info["truedyn_info"])

        surr_tune_result = tuning_data.get("surr_tune_result",None)

        tune_result = ControlTunerResult(inc_cfg = inc_cfg,
                cfgs = cfgs,
                inc_cfgs = inc_cfgs,
                costs = costs,
                inc_costs = inc_costs,
                truedyn_costs = truedyn_costs,
                inc_truedyn_costs = inc_truedyn_costs,
                surr_infos = surr_infos,
                truedyn_infos = truedyn_infos,
                surr_tune_result = surr_tune_result)

        return tune_result

    def run(self, controller, tasks, trajs, n_iters, rng, truedyn=None, 
            surrogate_tune_iters=100, eval_timeout=600, output_dir=None, restore_dir=None,
            save_all_controllers=False, use_default_initial_design=True,
            debug_return_evaluator=False, max_eval_jobs=1):
        """
        Run tuning.

        Parameters
        ----------
        controller : Controller
            Controller to tune.  Can be an AutoSelectController or a Controller
            with a manually-configured configuration space.

        tasks : Task or [Task]
            Tasks which specify the tuning problem.

        trajs : List of Trajectory
            Trajectory training set.

        n_iters : int
            Number of tuning iterations

        rng : numpy.random.Generator
            RNG to use for tuning.

        truedyn : obs, ctrl -> obs
            True dynamics function. If passed, the true dynamics cost
            will be evaluated for each iteration in addition to the surrogate
            cost. However, this information will not be used for tuning.

        surrogate_tune_iters : int
            Number of iterations to use for surrogate tuning. Used for "autotune"
            and "autoselect" modes. Default is 100

        eval_timeout : float
            Maximum number of seconds to allow for tuning of a single configuration.
            Default is 600.

        output_dir : str
            Output directory to store intermediate tuning information.  If None,
            a filename will be automatically selected.

        restore_dir : str
            To resume a previous tuning process, pass the output_dir for the previous
            tuning process here. 

        Returns
        -------
        controller : Controller
            Final tuned controller.  (If deploying, you should run controller.build(trajs) to
            re-train on the whole dataset)

        tune_result : PipelineTuneResult
            Additional tuning information.
        """

        # Initialize output directories
        if output_dir is None:
            output_dir = "autompc-output_" + datetime.datetime.now().isoformat(timespec="seconds")
        run_dir = os.path.join(output_dir, 
            "run_{}".format(int(1000.0*datetime.datetime.utcnow().timestamp())))
        smac_dir = os.path.join(run_dir, "smac")
        eval_result_dir = os.path.join(run_dir, "eval_results")

        if restore_dir:
            restore_run_dir = self._get_restore_run_dir(restore_dir)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if os.path.exists(run_dir):
            raise Exception("Run directory already exists")
        os.mkdir(run_dir)

        if not os.path.exists(smac_dir):
            os.mkdir(smac_dir)
        if not os.path.exists(eval_result_dir):
            os.mkdir(eval_result_dir)
        if not os.path.exists(os.path.join(smac_dir, "run_1")):
            os.mkdir(os.path.join(smac_dir, "run_1"))
        if save_all_controllers:
            controller_save_dir = os.path.join(run_dir, "saved_controllers")
            if not os.path.exists(controller_save_dir):
                os.mkdir(controller_save_dir)
        else:
            controller_save_dir = None
        
        if isinstance(tasks, Task):
            tasks = [tasks]

        if not restore_dir:
            controller = controller.clone()
            controller.set_ocp(tasks[0].get_ocp())
            tuning_data = self._get_tuning_data(controller, tasks, trajs, truedyn, rng, surrogate_tune_iters)
            with open(os.path.join(run_dir, "tuning_data.pkl"), "wb") as f:
                pickle.dump(tuning_data, f)
        else:
            self._copy_restore_data(restore_run_dir, run_dir)
            with open(os.path.join(run_dir, "tuning_data.pkl"), "rb") as f:
                tuning_data = pickle.load(f)
            #surrogate_tune_result = tuning_data["surr_tune_result"]

        eval_cfg = CfgRunner(run_dir=run_dir, timeout=eval_timeout,
            controller_save_dir=controller_save_dir, eval_result_dir=eval_result_dir,
            max_eval_jobs=max_eval_jobs)

        if debug_return_evaluator:
            return eval_cfg
        smac_rng = np.random.RandomState(seed=rng.integers(1 << 31))
        scenario = Scenario({"run_obj" : "quality",
                             "runcount-limit" : n_iters,
                             "cs" : controller.get_config_space(),
                             "deterministic" : "true",
                             "limit_resources" : False,
                             "abort_on_first_run_crash" : False,
                             "save_results_instantly" : True,
                             "output_dir" : smac_dir
                             })

        if not use_default_initial_design:
            initial_design = RandomConfigurations
        else:
            initial_design = None

        if not restore_dir:
            smac = SMAC4HPO(scenario=scenario, rng=smac_rng,
                    initial_design=initial_design,
                    tae_runner=eval_cfg,
                    run_id = 1
                    )
        else:
            runhistory, stats, incumbent = self._load_smac_restore_data(restore_run_dir, scenario)
            smac = SMAC4HPO(scenario=scenario, rng=smac_rng,
                    initial_design=initial_design,
                    tae_runner=eval_cfg,
                    run_id = 1,
                    runhistory=runhistory,
                    stats=stats,
                    restore_incumbent=incumbent
                    )  

        inc_cfg = smac.optimize()

        # Generate final model and controller
        tuned_controller = controller.clone()
        tuned_controller.set_ocp(tasks[0].get_ocp())
        tuned_controller.set_config(inc_cfg)
        tuned_controller.build(tuning_data["sysid_trajs"])

        tune_result = self._get_tune_result(tuning_data, smac.runhistory)

        return tuned_controller, tune_result



class CfgRunner:
    def __init__(self, tuning_data=None, run_dir=None, timeout=None, log_file_name=None,
                    controller_save_dir=None, eval_result_dir=None, max_eval_jobs=1):
        if tuning_data is None and run_dir is None:
            raise ValueError("Either tuning_data or run_dir must be passed")
        self.tuning_data = tuning_data
        self.run_dir = run_dir
        self.timeout = timeout
        if log_file_name is None and run_dir is not None:
            log_file_name = os.path.join(run_dir, "log.txt")
        self.log_file_name = log_file_name
        self.eval_number = 0
        self.controller_save_dir = controller_save_dir
        self.eval_result_dir = eval_result_dir
        self.max_eval_jobs = max_eval_jobs

    def get_tuning_data(self):
        if not self.tuning_data is None:
            return self.tuning_data
        with open(os.path.join(self.run_dir, "tuning_data.pkl"), "rb") as f:
            return pickle.load(f)

    def __call__(self, cfg):
        self.eval_number += 1
        if self.timeout is None: 
            result = self.run(cfg)
        else:
            #p = multiprocessing.Process(target=self.run_mp, args=(cfg,))
            ctx = multiprocessing.get_context("spawn")
            q = ctx.Queue()
            p = ctx.Process(target=self.run_mp, args=(cfg, q))
            start_time = time.time()
            p.start()
            timeout = False
            while p.is_alive():
                if time.time() - start_time > self.timeout:
                    timeout = True
                    break
                try:
                    result = q.get(block=True, timeout=10.0)
                    break
                except queue.Empty:
                    continue
            p.join(timeout=1)
            if timeout:
                print("CfgRunner: Evaluation timed out")
                p.terminate()
                return np.inf, dict()
            if p.exitcode != 0:
                print("CfgRunner: Exception during evaluation")
                print("Exit code: ", p.exitcode)
                return np.inf, dict()
        
        result['surr_info'] = [trial_to_json(info) for info in result['surr_info']]
        if 'truedyn_info' in result:
            result['truedyn_info'] = [trial_to_json(info) for info in result['truedyn_info']]
        return result['surr_cost'],dict(result)

    def run_mp(self, cfg, q):
        if not self.log_file_name is None:
            with open(self.log_file_name, "a") as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    try:
                        result = self.run(cfg)
                    except Exception as e:
                        print("Exception raised: \n", str(e))
                        raise e
                    print("Putting Result... ", end="")
                    q.put(result)
                    print("Done.")
        else:
            result = self.run(cfg)
            q.put(result)

    # def dispatch_eval_jobs(self, controller, evaluator):
    #     ctx = multiprocessing.get_context("spawn")
    #     active_qs = []
    #     active_ps = []
    #     job_idx = 0
    #     for _ in range(self.max_eval_jobs):
    #     with open 

    def run(self, cfg):
        print("\n>>> ", datetime.datetime.now(), "> Evaluating Cfg: \n", cfg)
        tuning_data = self.get_tuning_data()
        controller = tuning_data["controller"].clone()
        sysid_trajs = tuning_data["sysid_trajs"]
        control_evaluator = tuning_data["control_evaluator"]  # type : ControlEvaluator
        truedyn_evaluator = tuning_data["truedyn_evaluator"]  # type : ControlEvaluator
        performance_metric = tuning_data["performance_metric"]  # type : ControlPerformanceMetric
        info = dict()

        controller.set_config(cfg)
        controller.build(sysid_trajs)
        trajs = control_evaluator(controller)
        performance = performance_metric(trajs)
        info["surr_cost"] = performance
        info["surr_info"] = trajs
        if truedyn_evaluator is not None:
            trajs = truedyn_evaluator(controller)
            performance = performance_metric(trajs)
            info["truedyn_cost"] = performance
            info["truedyn_info"] = trajs
        
        if self.controller_save_dir:
            controller_save_fn = os.path.join(self.controller_save_dir, "controller_{}.pkl".format(self.eval_number))
            with open(controller_save_fn, "wb") as f:
                pickle.dump(controller, f)

        # if not self.log_file_name is None:
        #     sys.stdout.close()
        #     sys.stderr.close()
        return info
