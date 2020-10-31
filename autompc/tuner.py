# Created by William Edwards (wre2@illinois.edu)

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC
import copy

from .cs_utils import add_configuration_space

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

from pdb import set_trace


class ModelTuner:
    def __init__(self, system, evaluator):
        self.system = system
        self.evaluator = evaluator
        self.models = []

    def add_model(self, model, cs=None):
        if cs is None:
            cs = model.get_configuration_space(self.system)
        self.models.append((model, cs))

    def _get_model_cfg(self, cfg_combined):
        for model, cs in self.models:
            if model.__name__ != cfg_combined["model"]:
                continue
            cfg = cs.get_default_configuration()
            prefix = "_" + model.__name__ + ":"
            for key, val in cfg_combined.get_dictionary().items():
                if key[:len(prefix)] == prefix:
                    cfg[key.split(":")[1]] = val
            return model, cfg

    def _evaluate(self, cfg_combined):
        print("Evaluating cfg:")
        print(cfg_combined)
        model, cfg = self._get_model_cfg(cfg_combined)
        return self.evaluator(model, cfg)[0]



    def run(self, rng, runcount_limit=10, n_jobs = 1): 
        # Construct configuration space
        cs_combined = CS.ConfigurationSpace()

        model_choice = CSH.CategoricalHyperparameter("model",
                choices=[model.__name__ for model, _ in self.models])
        cs_combined.add_hyperparameter(model_choice)
        for model, cs in self.models:
            model_name = model.__name__
            add_configuration_space(cs_combined, "_" + model_name, 
                    cs, parent_hyperparameter={"parent" : model_choice, 
                        "value" : model.__name__})

        scenario = Scenario({"run_obj": "quality",  
                             "runcount-limit": runcount_limit,  
                             "cs": cs_combined,  
                             "deterministic": "true"
                             })

        smac = SMAC4HPO(scenario=scenario, rng=rng,
                tae_runner=self._evaluate, n_jobs=n_jobs)

        incumbent = smac.optimize()

        ret_value = dict()
        ret_value["incumbent"] = incumbent
        model, inc_cfg = self._get_model_cfg(incumbent)
        ret_value["inc_cfg"] = inc_cfg
        inc_cost = float("inf")
        inc_costs = []
        evaluated_costs = []
        evaluated_cfgs = []
        costs_and_config_ids = []
        for key, val in smac.runhistory.data.items():
            if val.cost < inc_cost:
                inc_cost = val.cost
            inc_costs.append(inc_cost)
            evaluated_costs.append(val.cost)
            evaluated_cfgs.append(smac.runhistory.ids_config[key.config_id])
            costs_and_config_ids.append((val.cost, key.config_id))
        ret_value["inc_costs"] = inc_costs
        ret_value["evaluated_costs"] = evaluated_costs
        ret_value["evaluated_cfgs"] = evaluated_cfgs
        costs_and_config_ids.sort()
        #if len(costs_and_config_ids) >= 5:
        #    top_five = [(smac.runhistory.ids_config[cfg_id], cost) for cost, cfg_id 
        #        in costs_and_config_ids[:5]]
        #else:
        #    top_five = [(smac.runhistory.ids_config[cfg_id], cost) for cost, cfg_id 
        #        in costs_and_config_ids]
        #ret_value["top_five"] = top_five

        return ret_value


