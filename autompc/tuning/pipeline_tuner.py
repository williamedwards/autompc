
# Standard library includes
from collections import namedtuple
import pickle
import multiprocessing
import contextlib
import datetime
import os, sys, glob, shutil

# Internal project includes
from .. import zeros, MPCCompatibilityError
from ..utils import simulate
from ..evaluation import HoldoutModelEvaluator
from .model_tuner import ModelTuner
from ..sysid import MLPFactory, SINDyFactory, ApproximateGPModelFactory, ARXFactory, KoopmanFactory

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
    "surr_trajs", "truedyn_trajs", "surr_tune_result"])
"""
PipelineTuneREsult contains information about a tuning process.

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


#@pynisher.enforce_limits(wall_time_in_s=300, grace_period_in_s=10)
class CfgEvaluator:
    def __init__(self, tuning_data=None, run_dir=None, timeout=None, log_file_name=None,
                    controller_save_dir=None):
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
        p.start()
        p.join(timeout=self.timeout)
        # p.join()
        if p.exitcode is None:
            print("CfgEvaluator: Evaluation timed out")
            p.terminate()
            return np.inf, dict()
        if p.exitcode != 0:
            print("CfgEvaluator: Exception during evaluation")
            print("Exit code: ", p.exitcode)
            return np.inf, dict()
        else:
            result = q.get()
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
        else:
            result = self.run(cfg)
        q.put(result)

    def run(self, cfg):
        print("\n>>> ", datetime.datetime.now(), "> Evaluating Cfg: \n", cfg)
        tuning_data = self.get_tuning_data()
        pipeline = tuning_data["pipeline"]
        task = tuning_data["task"]
        surrogate = tuning_data["surrogate"]
        sysid_trajs = tuning_data["sysid_trajs"]
        truedyn = tuning_data["truedyn"]
        info = dict()
        try:
            controller, cost, model = pipeline(cfg, sysid_trajs)
            print("Simulating Surrogate Trajectory: ")
            try:
                controller.reset()
                if task.has_num_steps():
                    surr_traj = simulate(controller, task.get_init_obs(),
                        task.term_cond, sim_model=surrogate, 
                        ctrl_bounds=task.get_ctrl_bounds(),
                        max_steps=task.get_num_steps())
                else:
                    surr_traj = simulate(controller, task.get_init_obs(),
                        ctrl_bounds=task.get_ctrl_bounds(),
                        term_cond=task.term_cond, sim_model=surrogate)
                cost = task.get_cost()
                surr_cost = cost(surr_traj)
                print("Surrogate Cost: ", surr_cost)
                print("Surrogate Final State: ", surr_traj[-1].obs)
                info["surr_cost"] = surr_cost
                info["surr_traj"] = (surr_traj.obs.tolist(), surr_traj.ctrls.tolist())
            except np.linalg.LinAlgError:
                surr_cost = np.inf
                info["surr_cost"] = surr_cost
                info["surr_traj"] = None
            
            if not truedyn is None:
                print("Simulating True Dynamics Trajectory")
                controller, _, _ = pipeline(cfg, sysid_trajs, model=model)
                controller.reset()
                if task.has_num_steps():
                    truedyn_traj = simulate(controller, task.get_init_obs(),
                       task.term_cond, dynamics=truedyn, max_steps=task.get_num_steps())
                else:
                    truedyn_traj = simulate(controller, task.get_init_obs(),
                       task.term_cond, dynamics=truedyn)
                truedyn_cost = cost(truedyn_traj)
                print("True Dynamics Cost: ", truedyn_cost)
                print("True Dynamics Final State: ", truedyn_traj[-1].obs)
                info["truedyn_cost"] = truedyn_cost
                info["truedyn_traj"] = (truedyn_traj.obs.tolist(), 
                        truedyn_traj.ctrls.tolist())
            
            if self.controller_save_dir:
                controller_save_fn = os.path.join(self.controller_save_dir, "controller_{}.pkl".format(self.eval_number))
                with open(controller_save_fn, "wb") as f:
                    pickle.dump(controller, f)
        except MPCCompatibilityError:
            surr_cost = np.inf

        if not self.log_file_name is None:
            sys.stdout.close()
            sys.stderr.close()

        return surr_cost, info

class PipelineTuner:
    """
    This class tunes SysID+MPC pipelines.
    """
    def __init__(self, surrogate_mode="defaultcfg", surrogate_factory=None, surrogate_split=None, surrogate_cfg=None,
            surrogate_evaluator=None, surrogate_tune_holdout=0.25, surrogate_tune_metric="rmse"):
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

    def _get_surrogate(self, pipeline, trajs, rng, surrogate_tune_iters):
        surrogate_tune_result = None
        if self.surrogate_mode == "defaultcfg":
            surrogate_cs = self.surrogate_factory.get_configuration_space()
            surrogate_cfg = surrogate_cs.get_default_configuration()
            surrogate = self.surrogate_factory(surrogate_cfg, trajs)
        elif self.surrogate_mode == "fixedcfg":
            surrogate = self.surrogate_factory(self.surrogate_cfg, trajs)
        elif self.surrogate_mode == "autotune":
            evaluator = self.surrogate_evaluator
            if evaluator is None:
                evaluator = HoldoutModelEvaluator(
                        holdout_prop = self.surrogate_tune_holdout,
                        metric = self.surrogate_tune_metric,
                        system=pipeline.system,
                        trajs=trajs,
                        rng=rng)
            model_tuner = ModelTuner(pipeline.system, evaluator) 
            model_tuner.add_model_factory(self.surrogate_factory)
            surrogate, surrogate_tune_result = model_tuner.run(rng, n_iters=surrogate_tune_iters) 
        elif self.surrogate_mode == "autoselect":
            evaluator = self.surrogate_evaluator
            if evaluator is None:
                evaluator = HoldoutModelEvaluator(
                        holdout_prop = self.surrogate_tune_holdout,
                        metric = self.surrogate_tune_metric,
                        system=pipeline.system,
                        trajs=trajs,
                        rng=rng)
            model_tuner = ModelTuner(pipeline.system, evaluator) 
            for factory in autoselect_factories:
                model_tuner.add_model_factory(factory(pipeline.system))
            surrogate, surrogate_tune_result = model_tuner.run(rng, n_iters=surrogate_tune_iters) 
        return surrogate, surrogate_tune_result

    def _get_tuning_data(self, pipeline, task, trajs, truedyn, rng, surrogate, surrogate_tune_iters):
        # Run surrogate training
        if surrogate is None:
            surr_size = int(self.surrogate_split * len(trajs))
            shuffled_trajs = trajs[:]
            rng.shuffle(shuffled_trajs)
            surr_trajs = shuffled_trajs[:surr_size]
            sysid_trajs = shuffled_trajs[surr_size:]

            surrogate, surr_tune_result = self._get_surrogate(pipeline, surr_trajs, rng, surrogate_tune_iters)
        else:
            sysid_trajs = trajs
            surr_tune_result = None

        tuning_data = dict()
        tuning_data["pipeline"] = pipeline
        tuning_data["surrogate"] = surrogate
        tuning_data["task"] = task
        tuning_data["sysid_trajs"] = sysid_trajs
        tuning_data["surr_trajs"] = surr_trajs
        tuning_data["truedyn"] = truedyn

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

    def run(self, pipeline, task, trajs, n_iters, rng, surrogate=None, truedyn=None, 
            surrogate_tune_iters=100, eval_timeout=600, output_dir=None, restore_dir=None,
            save_all_controllers=False, debug_return_evaluator=False):
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

        if restore_dir:
            restore_run_dir = self._get_restore_run_dir(restore_dir)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if os.path.exists(run_dir):
            raise Exception("Run directory already exists")
        os.mkdir(run_dir)

        if not os.path.exists(smac_dir):
            os.mkdir(smac_dir)
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

        eval_cfg = CfgEvaluator(run_dir=run_dir, timeout=eval_timeout,
            controller_save_dir=controller_save_dir)

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

        if not restore_dir:
            smac = SMAC4HPO(scenario=scenario, rng=smac_rng,
                    initial_design=RandomConfigurations,
                    tae_runner=eval_cfg,
                    run_id = 1
                    )
        else:
            runhistory, stats, incumbent = self._load_smac_restore_data(restore_run_dir, scenario)
            smac = SMAC4HPO(scenario=scenario, rng=smac_rng,
                    initial_design=RandomConfigurations,
                    tae_runner=eval_cfg,
                    run_id = 1,
                    runhistory=runhistory,
                    stats=stats,
                    restore_incumbent=incumbent
                    )  

        inc_cfg = smac.optimize()

        cfgs, inc_cfgs, costs, inc_costs, truedyn_costs, inc_truedyn_costs, surr_trajs,\
                truedyn_trajs =  [], [], [], [], [], [], [], []
        inc_cost = float("inf")

        for key, val in smac.runhistory.data.items():
            cfg = smac.runhistory.ids_config[key.config_id]
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
            if val.additional_info["surr_traj"] is not None:
                surr_obs, surr_ctrls = val.additional_info["surr_traj"]
                surr_traj = zeros(pipeline.system, len(surr_obs))
                surr_traj.obs[:] = surr_obs
                surr_traj.ctrls[:] = surr_ctrls
                surr_trajs.append(surr_traj)
            else:
                surr_trajs.append(None)
            if "truedyn_cost" in val.additional_info:
                inc_truedyn_costs.append(inc_truedyn_cost)
                truedyn_costs.append(val.additional_info["truedyn_cost"])
                truedyn_obs, truedyn_ctrls = val.additional_info["truedyn_traj"]
                truedyn_traj = zeros(pipeline.system, len(truedyn_obs))
                truedyn_traj.obs[:] = truedyn_obs
                truedyn_traj.ctrls[:] = truedyn_ctrls
                truedyn_trajs.append(truedyn_traj)

        tune_result = PipelineTuneResult(inc_cfg = inc_cfg,
                cfgs = cfgs,
                inc_cfgs = inc_cfgs,
                costs = costs,
                inc_costs = inc_costs,
                truedyn_costs = truedyn_costs,
                inc_truedyn_costs = inc_truedyn_costs,
                surr_trajs = surr_trajs,
                truedyn_trajs = truedyn_trajs,
                surr_tune_result = surr_tune_result)

        # Generate final model and controller
        controller, cost, model = pipeline(inc_cfg, task, sysid_trajs)

        return controller, tune_result
