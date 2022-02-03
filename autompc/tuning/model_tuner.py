# Created by William Edwards (wre2@illinois.edu)

from collections import namedtuple

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC
import copy

from ..utils.cs_utils import add_configuration_space

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    Hyperparameter,
    Constant,
    FloatHyperparameter,
)
from ConfigSpace.conditions import (
    ConditionComponent,
    AbstractCondition,
    AbstractConjunction,
    EqualsCondition,
)
from ConfigSpace.forbidden import (
    AbstractForbiddenComponent,
    AbstractForbiddenClause,
    AbstractForbiddenConjunction,
)

import numpy as np

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
    The ModelTuner class is used for tuning system ID models based
    on prediction accuracy.
    """
    def __init__(self, system, evaluator):
        """
        Parameters
        ----------
        system : System
            System for which models will be tuned
        evaluator : ModelEvaluator
            This evaluator object will be used to asses model
            configurations
        """
        self.system = system
        self.evaluator = evaluator
        self.models_and_cs = []

    def add_model(self, model, cs=None):
        """
        Add a model which is an option for tuning.
        Multiple model factories can be added and the tuner
        will select between them.

        Parameters
        ----------
        model : Model
            Model to be considered for tuning.

        cs : ConfigurationSpace
            Configuration space for model. This only needs to be
            passed if the configuration space is customized, otherwise
            it will be derived from the model_factory.
        """
        if cs is None:
            cs = model.get_config_space()
        self.models_and_cs.append((model, cs))

    def _get_model_cfg(self, cfg_combined):
        for model, cs in self.models_and_cs:
            if model.name != cfg_combined["model"]:
                continue
            cfg = cs.get_default_configuration()
            prefix = "_" + model.name + ":"
            for key, val in cfg_combined.get_dictionary().items():
                if key[:len(prefix)] == prefix:
                    cfg[key.split(":", 1)[1]] = val
            return model, cfg

    def _evaluate(self, cfg_combined):
        print("Evaluating Cfg:")
        print(cfg_combined)
        model, cfg = self._get_model_cfg(cfg_combined)
        value = self.evaluator(model, cfg)
        print("Model Score ", value)
        return value



    def run(self, rng, n_iters=10): 
        """
        Run tuning process

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator used for tuning

        n_iters : int
            Number of tuning iterations to run

        Returns
        -------
        model : Model
            Resulting model

        tune_result : ModelTuneResult
            Additional information from tuning process
        """
        # Construct configuration space
        cs_combined = CS.ConfigurationSpace()

        model_choice = CSH.CategoricalHyperparameter("model",
                choices=[model.name for model, _ 
                    in self.models_and_cs])
        cs_combined.add_hyperparameter(model_choice)
        for model, cs in self.models_and_cs:
            model_name = model.name
            add_configuration_space(cs_combined, "_" + model_name, 
                    cs, parent_hyperparameter={"parent" : model_choice, 
                        "value" : model_name})

        smac_rng = np.random.RandomState(rng.integers(1 << 31))
        scenario = Scenario({"run_obj": "quality",  
                             "runcount-limit": n_iters,  
                             "cs": cs_combined,  
                             "deterministic": "true",
                             "limit_resources" : False
                             })

        smac = SMAC4HPO(scenario=scenario, rng=smac_rng,
                tae_runner=self._evaluate)

        incumbent = smac.optimize()

        ret_value = dict()
        inc_cost = float("inf")
        inc_costs = []
        evaluated_costs = []
        evaluated_cfgs = []
        inc_cfgs = []
        costs_and_config_ids = []
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

        model, inc_cfg = self._get_model_cfg(incumbent)
        final_model = model.clone()
        final_model.set_config(inc_cfg)
        final_model.train(self.evaluator.trajs)

        return final_model, tune_result
