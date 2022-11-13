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

    @property
    def restore(self):
        return self.restore_dir is not None

    def restore_cfg_evaluator(self) -> CfgEvaluator:
        restore_run_dir = self._get_restore_run_dir()
        with open(os.path.join(restore_run_dir, "cfg_evaluator.pkl"), "rb") as f:
            cfg_evaluator = pickle.load(f)
        return cfg_evlauator

    def _init_output_directories(self, cfg_evaluator):
        # Construct output paths
        if self.output_dir is None:
            output_dir = Path("autompc-output_" + datetime.datetime.now().isoformat(timespec="seconds"))
        else:
            output_dir = Path(self.output_dir)
        run_dir = output_dir / "run_{}".format(int(1000.0*datetime.datetime.utcnow().timestamp()))
        smac_dir = run_dir / "smac"
        eval_result_dir = run_dir / "eval_results"

        # Create output directories if needed
        if run_dir.exists():
            raise Exception("Run directory already exists")
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir()
        eval_result_dir.mkdir()
        smac_dir.mkdir(exist_ok=True)
        (smac_dir / "run_1").mkdir(exist_ok=True)

        # Save config evaluator
        with open(run_dir / "cfg_evaluator.pkl", "wb") as f:
            pickle.dump(cfg_evaluator, f)

        return run_dir, smac_dir, eval_result_dir
    
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

    def run(self, cs: ConfigurationSpace, cfg_evaluator: CfgEvaluator, n_iters: int, rng: np.random.Generator, eval_timeout: float):
        run_dir, smac_dir, eval_result_dir = self._init_output_directories(cfg_evaluator)

        smac_rng = np.random.RandomState(seed=rng.integers(1 << 31))
        scenario = Scenario({"run_obj" : "quality",
                             "runcount-limit" : n_iters,
                             "cs" : cs,
                             "deterministic" : "true",
                             "limit_resources" : False,
                             "abort_on_first_run_crash" : False,
                             "save_results_instantly" : True,
                             "output_dir" : smac_dir
                             })

        if not self.use_default_initial_design:
            initial_design = RandomConfigurations
        else:
            initial_design = None

        cfg_runner = CfgRunner(run_dir=run_dir, eval_result_dir=eval_result_dir, timeout=eval_timeout, log_file_name=run_dir/"log.txt")

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
    def __init__(self, run_dir, eval_result_dir, timeout=None, log_file_name=None):
        self.run_dir = run_dir
        self.timeout = timeout
        self.log_file_name = log_file_name
        self.eval_result_dir = eval_result_dir
        self.eval_number = 0

    def get_cfg_evaluator(self):
        with open(self.run_dir / "cfg_evaluator.pkl", "rb") as f:
            return pickle.load(f)

    def __call__(self, cfg):
        self.eval_number += 1

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
            return np.inf, dict()
        if p.exitcode != 0:
            print("CfgRunner: Exception during evaluation")
            print("Exit code: ", p.exitcode)
            return np.inf, dict()

        return result


    def run_mp(self, cfg, q):
        cfg_evaluator = self.get_cfg_evaluator()
        if not self.log_file_name is None:
            with open(self.log_file_name, "a") as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                    try:
                        result = cfg_evaluator(cfg)
                    except Exception as e:
                        print("Exception raised: \n", str(e))
                        raise e
                    print("Putting result...")
                    q.put(result)
                    print("Done.")
        else:
            result = cfg_evaluator(cfg)
            q.put(result)