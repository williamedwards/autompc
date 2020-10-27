# Created by William Edwards (wre2@illinois.edu)

import math, time
from pdb import set_trace
import numpy as np

from ..evaluator import Evaluator, CachingPredictor
from .. import utils

class FixedSetEvaluator(Evaluator):
    def __init__(self, *args, training_trajs, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_set = training_trajs[:]
        self.holdout = []
        for traj in self.trajs:
            if not traj in training_trajs:
                self.holdout.append(traj)

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
        primary_metric_values = np.zeros(len(self.holdout))
        secondray_metric_values = np.zeros((len(self.holdout),
            len(self.secondary_metrics)))
        graphs = [grapher(model, configuration) for grapher in self.graphers]
        for i, traj in enumerate(self.holdout):
            predictor = CachingPredictor(traj, m)
            primary_metric_values[i] = self.primary_metric(predictor, traj)
            for j, metric in enumerate(self.secondary_metrics):
                secondary_metric_values[i, j] = metric(predictor, traj)
            for graph in graphs:
                graph.add_traj(predictor, traj)

        need_training_eval = False
        for graph in graphs:
            if graph.need_training_eval:
                need_training_eval = True
                break

        if need_training_eval:
            for traj in self.training_set:
                predictor = CachingPredictor(traj, m)
                for graph in graphs:
                    if graph.need_training_eval:
                        graph.add_traj(predictor, traj, training=True)

        primary_metric_value = self.primary_metric.accumulate(primary_metric_values)
        secondary_metric_value = np.array(len(self.secondary_metrics))

        for j, metric in enumerate(self.secondary_metrics):
            secondary_metric_value[j] = metric.accumulate(secondary_metric_values[:,j])
        print("Evaluation completed in {} sec".format(time.time() - eval_start))

        #print("k = {}, score = {}".format(m.k, primary_metric_value))
        print("CFG:")
        print(configuration)
        print("score = {}".format(primary_metric_value))

        if not ret_trained_model:
            return primary_metric_value, secondary_metric_value, graphs
        else:
            return primary_metric_value, secondary_metric_value, graphs, m
