# Created by William Edwards (wre2@illinois.edu)

from collections import namedtuple
from typing import Tuple,List,Union,Optional
import numpy as np

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_ac_facade import SMAC4AC
try:
    from smac.facade.smac_mf_facade import SMAC4MF
    SMAC4_MF_AVAILABLE = True
except ImportError:
    SMAC4_MF_AVAILABLE = False

import ConfigSpace as CS

from .model_evaluator import CrossValidationModelEvaluator, HoldoutModelEvaluator, ModelEvaluator
from ..system import System
from ..trajectory import Trajectory
from ..sysid.model import Model
from ..sysid.autoselect import AutoSelectModel

from pdb import set_trace

ModelTuneResult = namedtuple("ModelTuneResult", ["inc_cfg", "cfgs", 
    "inc_cfgs", "costs", "inc_costs"])
"""
The ModelTuneResult contains information about a tuning process.

.. py:attribute:: inc_cfg
    
    The final tuned configuration.

.. py:attribute:: cfgs

    List of configurations. The configuration evaluated at each
    tuning iteration.

.. py:attribute:: costs

    The cost (evaluation score) observed at each iteration of
    tuning iteration.

.. py:attribute:: inc_cfgs

    The incumbent (best found so far) configuration at each
    tuning iteration.

.. py:attribute:: inc_costs

    The incumbent score at each tuning iteration.
"""


class ModelTuner:
    """
    Used for tuning system ID models based on prediction accuracy.
    """
    def __init__(self, system : System, trajs : List[Trajectory], model : Optional[Model] = None,
                eval_holdout=0.25, eval_folds=3, eval_metric="rmse", eval_horizon=1, eval_quantile=None,
                evaluator : Optional[ModelEvaluator] = None,
                multi_fidelity=False, verbose=0):
        """
        Parameters
        ----------
        system : System
            System for which models will be tuned
        trajs : List[Trajectory]
            Trajectories on which to tune
        model : Model
            The model factory.  If None, will use an AutoSelectModel.
        eval_holdout : float
            If evaluator = None, and eval_folds <= 1, will use a HoldoutModelEvaluator with
            this fraction of a holdout set.
        eval_folds : int
            If evaluator = None, will use a CrossValidationModelEvaluator with this number
            of folds
        eval_metric : str or callable
            An evaluation metric, can be 'rmse', 'rmsmens', or a function (model,trajs,horizon) -> float
        eval_horizon : int
            The prediction horizon for evaluating model accuracy.
        eval_quantile : float or None
            Use a quantile-based performance metric.
        evaluator : ModelEvaluator
            The evaluator object used to assess model configurations. By default,
            will generate an evaluator using eval_holdout, eval_folds, and eval_metric.
        multi_fidelity : bool
            Whether to use the multi-fidelity SMAC
        verbose : int
            Whether to print messages. 1 = basic messages, 2 = detailed messages.
        """
        if model is None:
            model = AutoSelectModel(system)
        else:
            if system != model.system:
                raise ValueError("system and model.system must match")
        if evaluator is None:
            if eval_folds <= 1:
                evaluator = HoldoutModelEvaluator(trajs, eval_metric, horizon=eval_horizon, quantile=eval_quantile, holdout_prop=eval_holdout)
            else:
                evaluator = CrossValidationModelEvaluator(trajs, eval_metric, horizon=eval_horizon, quantile=eval_quantile, num_folds=eval_folds)
        else:
            evaluator.trajs = trajs
        
        self.system = system
        self.model = model           # type: Model
        self.evaluator = evaluator   # type: ModelEvaluator
        self.multi_fidelity = multi_fidelity
        self.verbose = verbose

    def _evaluate(self, cfg, seed=None, budget=None):
        if self.verbose:
            print("Evaluating Cfg:")
            print(cfg)
            print("Seed",seed,"budget",budget)
        self.model.set_config(cfg)
        self.model.set_train_budget(budget)
        value = self.evaluator(self.model)
        if self.verbose:
            print("Model Score ", value)
        return value

    def run(self, rng=None, n_iters=10, min_train_time=None, max_train_time=None,
        retrain_full=True) -> Tuple[Model,ModelTuneResult]: 
        """
        Run tuning process

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator used for tuning

        n_iters : int
            Number of tuning iterations to run
        
        min_train_time : float
            Initial amount of time allocated to train models. In multi-fidelity mode,
            default will use 10s.

        max_train_time : float
            Maximum amount of time allocated to train models. In multi-fidelity mode,
            default will use 180s.
        
        retrain_full : bool
            Whether we should retrain after tuning.

        Returns
        -------
        model : Model
            Resulting model

        tune_result : ModelTuneResult
            Additional information from tuning process.  Can access
            tune_result.inc_cfg to reconsruct the model.
        """
        if rng is None:
            rng = np.random.default_rng()
        cs = self.model.get_config_space()
        self.evaluator.rng = rng
        smac_rng = np.random.RandomState(rng.integers(1 << 31))
        scenario = Scenario({"run_obj": "quality",  
                             "runcount-limit": n_iters,  
                             "cs": cs,
                             "deterministic": "true",
                             'cutoff' : 600,
                             "limit_resources" : False
                             })

        if self.multi_fidelity and SMAC4_MF_AVAILABLE:
            #intensifier configuration
            min_train_time = min_train_time if min_train_time is not None else 10.0
            max_train_time = max_train_time if max_train_time is not None else 180.0
            intensifier_kwargs = {"initial_budget": min_train_time, "max_budget": max_train_time, "eta": 3}
            smac = SMAC4MF(scenario=scenario, rng=smac_rng,
                tae_runner=self._evaluate,
                intensifier_kwargs=intensifier_kwargs)
        else:
            if max_train_time is not None:
                self.model.set_train_budget(max_train_time)
            #smac = SMAC4HPO(scenario=scenario, rng=smac_rng,
            #        tae_runner=self._evaluate)
            smac = SMAC4AC(scenario=scenario, rng=smac_rng,
                    tae_runner=self._evaluate)
        
        incumbent = smac.optimize()

        inc_cost = float("inf")
        inc_costs = []
        evaluated_costs = []
        evaluated_cfgs = []
        inc_cfgs = []
        inc_cfg = None
        for key, val in smac.runhistory.data.items():
            cfg = smac.runhistory.ids_config[key.config_id]
            if val.cost < inc_cost:
                inc_cost = val.cost
                inc_cfg = cfg
            inc_costs.append(inc_cost)
            evaluated_costs.append(val.cost)
            evaluated_cfgs.append(cfg)
            inc_cfgs.append(inc_cfg)

        tune_result = ModelTuneResult(inc_cfg=inc_cfg,
                cfgs = evaluated_cfgs,
                costs = evaluated_costs,
                inc_costs = inc_costs,
                inc_cfgs = inc_cfgs)

        self.model.set_config(incumbent)
        if isinstance(self.model, AutoSelectModel):
            final_model = self.model.selected()
        else:
            final_model = self.model.clone()
        final_model.freeze_hyperparameters()
        if retrain_full:
            final_model.train(self.evaluator.trajs)

        return final_model, tune_result
