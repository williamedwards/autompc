import numpy as np
import math
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC
from tqdm import tqdm

from autompc.sysid.model import Model, ModelFactory

class ScheduledSamplingFactory(ModelFactory):
    def __init__(self, system, model_factory, decayfunction,
                ss_start_epsilon, ss_end_epsilon, ss_end_iter, # TODO: Figure out how to get iteration number from input model
                ss_exp_k,
                ss_invsig_k):
        self.model_factory = model_factory
        self.ss_end_iter = ss_end_iter
        if decayfunction == 'linear':
            self.decayfun = LinearDecay(ss_start_epsilon, ss_end_epsilon, ss_end_iter)
        elif decayfunction == 'exp':
            self.decayfun = ExponentialDecay(ss_exp_k)
        elif decayfunction == 'invsig':
            self.decayfun = InverseSigmoidDecay(ss_invsig_k)
        else:
            raise NotImplementedError("Currently supported decay functions: linear, exp, invsig")

    def get_configuration_space(self):
        orig_cs = self.model_factory.get_configuration_space() # edit directly
        decayfunction = CSH.CategoricalHyperparameter("decayfunction",
                choices=["linear", "exp", "invsig"], default_value="linear")
        ss_end_iter = CSH.Constant("ss_end_iter", value = self.ss_end_iter)
        start_epsilon = CSH.UniformFloatHyperparameter("ss_start_epsilon",
                lower = 0, upper = 1, default_value=1, log=False)
        end_epsilon = CSH.UniformFloatHyperparameter("ss_end_epsilon",
                lower = 0, upper = 1, default_value=0, log=False)
        exp_k = CSH.UniformFloatHyperparameter("ss_exp_k",
                lower = 0, upper = 1, default_value=.5, log=False)
        invsig_k = CSH.UniformFloatHyperparameter("ss_invsig_k",
                lower = 0, upper = 1500, default_value=120, log=False)
        start_epsilon_cond = CSC.EqualsCondition(start_epsilon, decayfunction,
                "linear")
        end_epsilon_cond = CSC.EqualsCondition(start_epsilon, decayfunction,
                "linear")
        end_epsilon_bound_cond = CSC.LessThanCondition(start_epsilon, end_epsilon, start_epsilon)
        exp_k_cond = CSC.EqualsCondition(exp_k, decayfunction,
                "exp")
        invsig_k_cond = CSC.EqualsCondition(invsig_k, decayfunction,
                "invsig")

        orig_cs.add_hyperparameters([decayfunction, ss_end_iter, start_epsilon, end_epsilon, exp_k, invsig_k])

        return orig_cs

    def __call__(self, cfg, train_trajs):
        model = ScheduledSamplingModel(self, self.model_factory, **cfg.get_dictionary())
        model.train(train_trajs)

class ScheduledSamplingModel(Model):
    def __init__(self, model_factory, exp_k=0.1, exp_c=0.5, **kwargs):
        self.model_factory = model_factory
        self.kwargs = kwargs
        model_cs = self.model_factory.get_configuration_space()
        model_cfg = CS.Configuration(model_cs, values=self.kwargs)
        

    def train(self, trajs):
        ...
        model = self.model_factory(model_cfg, skip_train_model=True)
        model.train(aug_dataset)

class LinearDecay:
   def __init__(self, start_epsilon, end_epsilon, end_iter):
       self.m = (start_epsilon - end_epsilon) / -end_iter
       self.b = start_epsilon

   def __call__(self, i):
       return self.m * i + self.b

class ExponentialDecay:
    def __init__(self, k):
       self.k = k

    def __call__(self, i):
       return self.k ^ i

class InverseSigmoidDecay:
    def __init__(self, k):
       self.k = k

    def __call__(self, i):
       return self.k / (self.k + math.exp(i / self.k))
        