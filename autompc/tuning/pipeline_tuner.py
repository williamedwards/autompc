
# Standard library includes
from collections import namedtuple
import pickle
import queue
import multiprocessing
import contextlib
import datetime
import time
import os, sys, glob, shutil

# Internal project includes
from .. import zeros, MPCCompatibilityError
from ..utils import simulate
from ..evaluation import HoldoutModelEvaluator
from .model_tuner import ModelTuner
from ..sysid import MLPFactory, SINDyFactory, ApproximateGPModelFactory, ARXFactory, KoopmanFactory
from .surrogate_evaluator import SurrogateEvaluator
from .true_dynamics_evaluator import TrueDynamicsEvaluator

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


PipelineTuneResult = namedtuple("PipelineTuneResult", ["inc_cfg", "cfgs", 
    "inc_cfgs", "costs", "inc_costs", "truedyn_costs", "inc_truedyn_costs", 
    "surr_dists", "truedyn_dists", "truedyn_eval_infos", "surr_eval_infos", 
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

autoselect_factories = [MLPFactory, SINDyFactory, ApproximateGPModelFactory,
        ARXFactory, KoopmanFactory]

class CfgRunner:
    def __init__(self, evaluation_quantile, tuning_data=None, run_dir=None, timeout=None, log_file_name=None,
                    controller_save_dir=None, eval_result_dir=None):
        self.evaluation_quantile = evaluation_quantile
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

    def get_tuning_data(self):
        if not self.tuning_data is None:
            return self.tuning_data
        with open(os.path.join(self.run_dir, "tuning_data.pkl"), "rb") as f:
            return pickle.load(f)

    def __call__(self, cfg):
        self.eval_number += 1
        if self.timeout is None:
            return self.run(cfg)

        #p = multiprocessing.Process(target=self.run_mp, args=(cfg,))
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=self.run_mp, args=(cfg, q))
        start_time = time.time()
        p.start()
        while p.is_alive():
            if time.time() - start_time > self.timeout:
                break
            try:
                result = q.get(block=True, timeout=10.0)
                break
            except queue.Empty:
                continue
        p.join(timeout=self.timeout)
        if p.exitcode is None:
            print("CfgRunner: Evaluation timed out")
            p.terminate()
            return np.inf, dict()
        if p.exitcode != 0:
            print("CfgRunner: Exception during evaluation")
            print("Exit code: ", p.exitcode)
            return np.inf, dict()
        else:
            #result = q.get()
            return result

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

    def run(self, cfg):
        print("\n>>> ", datetime.datetime.now(), "> Evaluating Cfg: \n", cfg)
        tuning_data = self.get_tuning_data()
        pipeline = tuning_data["pipeline"]
        task = tuning_data["task"]
        sysid_trajs = tuning_data["sysid_trajs"]
        control_evaluator = tuning_data["control_evaluator"]
        truedyn_evaluator = tuning_data["truedyn_evaluator"]
        info = dict()

        try:
            controller, _, _ = pipeline(cfg, sysid_trajs)
            surr_dist, surr_eval_info = control_evaluator(controller)
            surr_cost = surr_dist(self.evaluation_quantile)
            info["surr_cost"] = surr_cost
            info["surr_dist"] = str(surr_dist)
            info["surr_eval_info"] = surr_eval_info
            if not truedyn_evaluator is None:
                truedyn_dist, truedyn_eval_info = truedyn_evaluator(controller)
                truedyn_cost = truedyn_dist(self.evaluation_quantile)
                info["truedyn_cost"] = truedyn_cost
                info["truedyn_dist"] = str(truedyn_dist)
                info["truedyn_eval_info"] = truedyn_eval_info
            
            if self.controller_save_dir:
                controller_save_fn = os.path.join(self.controller_save_dir, "controller_{}.pkl".format(self.eval_number))
                with open(controller_save_fn, "wb") as f:
                    pickle.dump(controller, f)
        except MPCCompatibilityError:
            surr_cost = np.inf

        # if not self.log_file_name is None:
        #     sys.stdout.close()
        #     sys.stderr.close()

        return surr_cost, info

class PipelineTuner:
    """
    This class tunes SysID+MPC pipelines.
    """
    def __init__(self, surrogate_mode="defaultcfg", surrogate_factory=None, surrogate_split=None, 
            surrogate_cfg=None, evaluation_quantile=0.5,
            surrogate_evaluator=None, surrogate_tune_holdout=0.25, surrogate_tune_metric="rmse",
            control_evaluator_class=None, control_evaluator_kwargs=dict()): # TODO update docstring
        """
        Parameters
        ----------
        surrogate_mode : string
            Mode for selecting surrogate model. One of the following: "defaultcfg" - use the surrogate factories
            default configuration, "fixedcfg" - use the surrogate configuration passed by surrogate_cfg,
            "autotune" - automatically tune hyperparameters of surrogate, "autoselect" - automatically tune and
            select surrogate model, "pretrain" - Use an already trained surrogate which is passed when the tuning
            process is run.
        surrogate_factory : ModelFactory
            Factory for creating surrogate model. Required for all modes except for autoselect.
        surrogate_split : float
            Proportion of data to use for surrogate training.  Required for all modes except "pretrain"
        surrogate_cfg : Configuration
            Surrogate model config, required for "fixedcfg" mode
        surrogate_evaluator : ModelEvaluator
            Evaluator to use for surrogate tuning, used for "autoselect" and "autotune" modes. If not
            passed, will use HoldoutEvaluator with default arguments.
        surrogate_tune_holdout : float
            Proportion of data to hold out for surrogate tuning. Used for "autotune" and "autoselect" modes.
        surrogate_tune_metric : string
            Model metric to use for surrogate tuning.  Used for "autotune" and "autoselect" modes. See documentation
            of ModelEvaluator for more details.
        """
        self.surrogate_mode = surrogate_mode
        self.surrogate_factory = surrogate_factory
        self.surrogate_split = surrogate_split
        self.surrogate_cfg = surrogate_cfg
        self.surrogate_evaluator = surrogate_evaluator
        self.surrogate_tune_holdout = surrogate_tune_holdout
        self.surrogate_tune_metric = surrogate_tune_metric
        self.control_evaluator_class = control_evaluator_class
        self.control_evaluator_kwargs = control_evaluator_kwargs
        self.evaluation_quantile = evaluation_quantile

    def _get_tuning_data(self, pipeline, task, trajs, truedyn, rng, surrogate, surrogate_tune_iters):
        # Run surrogate training
        if surrogate is None:
            surr_size = int(self.surrogate_split * len(trajs))
            shuffled_trajs = trajs[:]
            rng.shuffle(shuffled_trajs)
            surr_trajs = shuffled_trajs[:surr_size]
            sysid_trajs = shuffled_trajs[surr_size:]

            if self.surrogate_factory is None:
                surrogate_factory = MLPFactory(pipeline.system)
            else:
                surrogate_factory = self.surrogate_factory

            control_eval_rng = np.random.default_rng(rng.integers(1 << 31))
            if self.control_evaluator_class is None:
                control_evaluator = SurrogateEvaluator(pipeline.system, task, surr_trajs,
                        surrogate_factory, rng=rng, surrogate_cfg = self.surrogate_cfg,
                        surrogate_mode=self.surrogate_mode, 
                        surrogate_tune_iters=surrogate_tune_iters, 
                        **self.control_evaluator_kwargs)
            else:
                control_evaluator = self.control_evaluator_class(pipeline.system, task,
                        surr_trajs, rng=control_eval_rng, **self.control_evaluator_kwargs)
        else:
            sysid_trajs = trajs
            control_evaluator = SurrogateEvaluator(pipeline.system, task, surr_trajs=None,
                    surrogate_factory=None, rng=control_eval_rng, surrogate=self.surrogate)

        if not truedyn is None:
            truedyn_evaluator = TrueDynamicsEvaluator(pipeline.system, task, trajs=None,
                    dynamics = truedyn)
        else:
            truedyn_evaluator = None

        tuning_data = dict()
        tuning_data["pipeline"] = pipeline
        tuning_data["control_evaluator"] = control_evaluator
        tuning_data["task"] = task
        tuning_data["sysid_trajs"] = sysid_trajs
        tuning_data["surr_trajs"] = surr_trajs
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
        cfgs, inc_cfgs, costs, inc_costs, truedyn_costs, inc_truedyn_costs, surr_dists, \
            truedyn_dists, surr_eval_infos, truedyn_eval_infos =  [], [], [], [], [], [], [], [], [], []
        inc_cost = float("inf")

        for key, val in runhistory.data.items():
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
            surr_eval_infos
            surr_dists.append(val.additional_info["surr_dist"])
            surr_eval_infos.append(val.additional_info["surr_eval_info"])
            if "truedyn_cost" in val.additional_info:
                inc_truedyn_costs.append(inc_truedyn_cost)
                truedyn_costs.append(val.additional_info["truedyn_cost"])
                truedyn_dists.append(val.additional_info["truedyn_dist"])
                truedyn_eval_infos.append(val.additional_info["truedyn_eval_info"])

        control_evaluator = tuning_data["control_evaluator"]
        if hasattr(control_evaluator, "surr_tune_result"):
            surr_tune_result = control_evaluator.surr_tune_result
        else:
            surr_tune_result = None

        tune_result = PipelineTuneResult(inc_cfg = inc_cfg,
                cfgs = cfgs,
                inc_cfgs = inc_cfgs,
                costs = costs,
                inc_costs = inc_costs,
                truedyn_costs = truedyn_costs,
                inc_truedyn_costs = inc_truedyn_costs,
                surr_dists = surr_dists,
                truedyn_dists = truedyn_dists,
                surr_eval_infos = surr_eval_infos,
                truedyn_eval_infos = truedyn_eval_infos,
                surr_tune_result = surr_tune_result)

        return tune_result

    def run(self, pipeline, task, trajs, n_iters, rng, surrogate=None, truedyn=None, 
            surrogate_tune_iters=100, eval_timeout=600, output_dir=None, restore_dir=None,
            save_all_controllers=False, use_default_initial_design=True,
            debug_return_evaluator=False): #TODO update docstring
        """
        Run tuning.

        Parameters
        ----------
        pipeline : Pipeline
            Pipeline to tune.

        task : Task
            Task specification to tune for

        trajs : List of Trajectory
            Trajectory training set.

        n_iters : int
            Number of tuning iterations

        rng : numpy.random.Generator
            RNG to use for tuning.

        surrogate : Model
            Surrogate model to use for tuning. Used for "pretrain" mode.

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
            Final tuned controller.

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

        if not restore_dir:
            tuning_data = self._get_tuning_data(pipeline, task, trajs, truedyn, rng, 
                surrogate, surrogate_tune_iters)
            with open(os.path.join(run_dir, "tuning_data.pkl"), "wb") as f:
                pickle.dump(tuning_data, f)
        else:
            self._copy_restore_data(restore_run_dir, run_dir)
            with open(os.path.join(run_dir, "tuning_data.pkl"), "rb") as f:
                tuning_data = pickle.load(f)
            #surrogate_tune_result = tuning_data["surr_tune_result"]

        eval_cfg = CfgRunner(self.evaluation_quantile, run_dir=run_dir, timeout=eval_timeout,
            controller_save_dir=controller_save_dir, eval_result_dir=eval_result_dir)

        if debug_return_evaluator:
            return eval_cfg

        smac_rng = np.random.RandomState(seed=rng.integers(1 << 31))
        scenario = Scenario({"run_obj" : "quality",
                             "runcount-limit" : n_iters,
                             "cs" : pipeline.get_configuration_space(),
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
        controller, cost, model = pipeline(inc_cfg, task, tuning_data["sysid_trajs"])

        tune_result = self._get_tune_result(tuning_data, smac.runhistory)

        return controller, tune_result
