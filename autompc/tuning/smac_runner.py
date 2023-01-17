from typing import Callable, Tuple, Dict, Optional, Any
from ConfigSpace import ConfigurationSpace, Configuration
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from pathlib import Path
from .data_store import DataStore
import multiprocessing
import contextlib
import datetime
import numpy as np
import pickle
import time
import queue
import sys


CfgEvaluator = Callable[[Configuration], Tuple[float, Dict[str, Any]]]

class SMACRunner:
    def __init__(self, output_dir: Optional[str] = None, restore_dir: Optional[str] = None, use_default_initial_design: bool = True):
        self.output_dir = output_dir
        self.restore_dir = restore_dir
        self.use_default_initial_design = use_default_initial_design

        self._init_output_directories()

    @property
    def restore(self):
        return self.restore_dir is not None

    def restore_cfg_evaluator(self) -> CfgEvaluator:
        restore_run_dir = self._get_restore_run_dir()
        with open(os.path.join(restore_run_dir, "cfg_evaluator.pkl"), "rb") as f:
            cfg_evaluator = pickle.load(f)
        return cfg_evlauator

    def _init_output_directories(self): #, cfg_evaluator):
        # Construct output paths
        if self.output_dir is None:
            output_dir = Path("autompc-output_" + datetime.datetime.now().isoformat(timespec="seconds"))
        else:
            output_dir = Path(self.output_dir)
        run_dir = output_dir / "run_{}".format(int(1000.0*datetime.datetime.utcnow().timestamp()))
        smac_dir = run_dir / "smac"
        eval_result_dir = run_dir / "eval_results"
        data_dir = run_dir / "datastore"

        # Create output directories if needed
        if run_dir.exists():
            raise Exception("Run directory already exists")
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir()
        eval_result_dir.mkdir()
        smac_dir.mkdir(exist_ok=True)
        (smac_dir / "run_1").mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)

        # Create data store
        self._data_store = DataStore(data_dir)

        self.run_dir = run_dir
        self.smac_dir = smac_dir
        self.eval_result_dir = eval_result_dir
    
    def _get_restore_run_dir(self):
        run_dirs = glob.glob(os.path.join(self.restore_dir, "run_*"))
        for run_dir in reversed(sorted(run_dirs)):
            if os.path.exists(os.path.join(run_dir, "smac", "run_1", "runhistory.json")):
                return run_dir
        raise FileNotFoundError("No valid restore files found")

    def _load_smac_restore_data(self, scenario):
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

    def _copy_restore_data(self):
        # Copy log
        old_log = os.path.join(restore_run_dir, "log.txt")
        new_log = os.path.join(new_run_dir, "log.txt")
        shutil.copy(old_log, new_log)
        # Copy smac trajectory information
        old_traj = os.path.join(restore_run_dir, "smac", "run_1", "traj_aclib2.json")
        new_traj = os.path.join(new_run_dir, "smac", "run_1", "traj_aclib2.json")
        shutil.copy(old_traj, new_traj)

    def get_data_store(self):
        return self._data_store

    def run(self, cs: ConfigurationSpace, cfg_evaluator: CfgEvaluator, n_iters: int, rng: np.random.Generator, eval_timeout: float):
        smac_rng = np.random.RandomState(seed=rng.integers(1 << 31))
        scenario = Scenario({"run_obj" : "quality",
                             "runcount-limit" : n_iters,
                             "cs" : cs,
                             "deterministic" : "true",
                             "limit_resources" : False,
                             "abort_on_first_run_crash" : False,
                             "save_results_instantly" : True,
                             "output_dir" : self.smac_dir
                             })

        if not self.use_default_initial_design:
            initial_design = RandomConfigurations
        else:
            initial_design = None

        cfg_runner = CfgRunner(
            cfg_evaluator=cfg_evaluator,
            timeout=eval_timeout, 
            log_file_name=self.run_dir/"log.txt",
            summary_log_file_name=self.run_dir/"summary_log.txt"
        )

        if not self.restore:
            smac = SMAC4HPO(scenario=scenario, rng=smac_rng,
                    initial_design=initial_design,
                    tae_runner=cfg_runner,
                    run_id = 1
                    )
        else:
            self._copy_restore_data()
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

        return inc_cfg, smac.runhistory

class CfgRunner:
    def __init__(self, cfg_evaluator, timeout=None, log_file_name=None, summary_log_file_name=None):
        self.cfg_evaluator = cfg_evaluator
        self.timeout = timeout
        self.log_file_name = log_file_name
        self.summary_log_file_name = summary_log_file_name

        self.summary_info = dict()

    def __call__(self, cfg):
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=self.run_mp, args=(cfg, q))
        start_time = time.time()

        p.start()
        timeout = False
        while p.is_alive():
            if self.timeout and time.time() - start_time > self.timeout:
                timeout = True
                break
            try:
                result = q.get(timeout=10, block=True)
                break
            except queue.Empty:
                pass

        p.join(timeout=10)

        if timeout:
            print("CfgRunner: Evaluation timed out")
            p.terminate()
            self.summary(None, None, timeout=True, exception=False)
            return np.inf, dict()
        if p.exitcode != 0:
            print("CfgRunner: Exception during evaluation")
            print("Exit code: ", p.exitcode)
            self.summary(None, None, timeout=False, exception=True)
            return np.inf, dict()

        cost, info = result

        if "truedyn_info" in info:
            truedyn_cost = info["truedyn_info"][0]["cost"]
        else:
            truedyn_cost = None 

        self.summary(cost, truedyn_cost, timeout=False, exception=False)

        return result

    def summary(self, cost, truedyn_cost, timeout, exception):
        # Compute number of evaluations, total, and those with
        # timeouts and exceptions
        def init_and_increment_counter(key, increment):
            if key not in self.summary_info:
                self.summary_info[key] = 0
            if increment:
                self.summary_info[key] += 1
        
        init_and_increment_counter("total_evaluations", True)
        init_and_increment_counter("timeouts", timeout)
        init_and_increment_counter("exceptions", exception)

        # Compute number of successful evalauations
        successful_evaluations = (
            self.summary_info["total_evaluations"] 
            - self.summary_info["timeouts"]
            - self.summary_info["exceptions"]
        )

        # Set incumbent costs
        if "inc_cost" not in self.summary_info:
            self.summary_info["inc_cost"] = np.inf

        if cost is not None and cost < self.summary_info["inc_cost"]:
            self.summary_info["inc_cost"] = cost

            if truedyn_cost is not None:           
                self.summary_info["inc_truedyn_cost"] = truedyn_cost

        # Generate summary string
        summary = f">>> {datetime.datetime.now()} > Summary:\n"
        summary += f"  Total Evaluations: {self.summary_info['total_evaluations']}\n"
        summary += f"  Successful Evaluations: {successful_evaluations}\n"
        summary += f"  Evaluations with Timeout: {self.summary_info['timeouts']}\n"
        summary += f"  Evaluations with Exception: {self.summary_info['exceptions']}\n"
        summary += f"  Incumbent Cost: {self.summary_info['inc_cost']}\n"
        if "inc_truedyn_cost" in self.summary_info:
            summary += f"  Incumbent True Dynamics Cost: {self.summary_info['inc_truedyn_cost']}\n"

        # Print summary
        with open(self.log_file_name, "a") as f:
            print(summary, file=f)

        if self.summary_log_file_name:
            with open(self.summary_log_file_name, "a") as f:
                print(summary, file=f)


    def run_mp(self, cfg, q):
        if not self.log_file_name is None:
            with open(self.log_file_name, "a") as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    try:
                        result = self.cfg_evaluator(cfg)
                    except Exception as e:
                        print("Exception raised: \n", str(e))
                        raise e
                    print("Putting result...")
                    q.put(result)
                    print("Done.")
        else:
            result = self.cfg_evaluator(cfg)
            q.put(result)