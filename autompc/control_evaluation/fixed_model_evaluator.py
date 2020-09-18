# Created by William Edwards (wre2@illinois.edu)

from .control_evaluator import ControlEvaluator
from ..tuner import ModelTuner
from ..utils import *
from ..cs_utils import *

class FixedModelEvaluator(ControlEvaluator):
    def __init__(self, system, task, metric, trajs, sim_model):
        super().__init__(system, task, metric)
        self.trajs = trajs
        self.sim_model = sim_model

    def __call__(self, pipeline):
        def eval_cfg(cfg):
            controller, model = pipeline(cfg, self.trajs)
            return self.metric(controller, self.sim_model)
        return eval_cfg

