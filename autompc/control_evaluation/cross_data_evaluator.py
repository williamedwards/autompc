# Created by William Edwards (wre2@illinois.edu)

from .control_evaluator import ControlEvaluator
from ..tuner import ModelTuner
from ..utils import *
from ..cs_utils import *

class CrossDataEvaluator(ControlEvaluator):
    def __init__(self, system, task, metric, ModelEvaluator, evaluator_kwargs,
            rng, training_trajs, validation_trajs, tuning_iters=10):
        super().__init__(system, task, metric)
        self.training_trajs = training_trajs
        self.validation_trajs = validation_trajs
        self.ModelEvaluator = ModelEvaluator
        self.evaluator_kwargs = evaluator_kwargs
        self.tuning_iters = tuning_iters
        self.rng = rng

    def __call__(self, pipeline, ret_model = False):
        Model = pipeline.Model
        evaluator = self.ModelEvaluator(self.system, self.validation_trajs,
                rng=self.rng, **self.evaluator_kwargs)
        tuner = ModelTuner(self.system, evaluator)
        tuner.add_model(Model)
        ret_value = tuner.run(rng=self.rng, runcount_limit = self.tuning_iters)
        inc_cfg = ret_value["inc_cfg"]

        sim_model = make_model(self.system, Model, inc_cfg)
        sim_model.train(self.validation_trajs)

        def eval_cfg(cfg):
            controller, model = pipeline(cfg, self.training_trajs)
            score = 0.0
            return self.metric(controller, sim_model)

        if not ret_model:
            return eval_cfg
        else:
            return eval_cfg, sim_model

