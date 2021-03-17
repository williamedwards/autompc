# Created by William Edwards (wre2@illinois.edu)

import math, time
from pdb import set_trace
import numpy as np

from ..evaluator import Evaluator, CachingPredictor
from .. import utils
from ..misc.model_metrics import *

class SimpleEvaluator:
    def __init__(self, system, training_set, validation_set, horiz):
        self.system = system
        self.training_set = training_set[:]
        self.validation_set = validation_set[:]
        self.horiz = horiz

    def __call__(self, model, configuration, ret_trained_model=False,
            trained_model=None, use_cuda=None):
        if trained_model is None:
            if use_cuda is None:
                m = utils.make_model(self.system, model, configuration)
            else:
                m = utils.make_model(self.system, model, configuration,
                        use_cuda=use_cuda)
            print("Entering training")
            train_start = time.time()
            m.train(self.training_set)
            print("Training completed in {} sec".format(time.time() - train_start))
        else:
            m = trained_model

        print("Entering evaluation.")
        eval_start = time.time()
        score = get_model_rmse(m, self.validation_set, horiz=self.horiz)

        set_trace()
        
        if not ret_trained_model:
            return score, None, None
        else:
            return score, None, None, m
