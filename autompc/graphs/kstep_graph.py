# Created by William Edwards

from pdb import set_trace

import numpy as np
import numpy.linalg as la

from ..evaluation.model_metrics import get_model_rmse

class KstepPredAccGraph:
    def __init__(self, system, trajs, kmax, logscale=False, metric="rmse"):
        self.kmax = kmax
        self.trajs = trajs
        self.logscale = logscale
        self.models = []
        self.labels = []

        if metric == "rmse":
            self.metric = get_model_rmse

    def add_model(self, model, label):
        self.models.append(model)
        self.labels.append(label)

    def get_rmses(self, model):
        return rmses, rmses_baseline, horizs

            
    def __call__(self, fig, ax):
        for model, label in zip(self.models, self.labels):
            rmses = [self.metric(model, self.trajs, horizon) 
                        for horizon in range(1, self.kmax)] 
            ax.plot(rmses, label=label)

        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel("Prediction Error")
        if self.logscale:
            ax.set_yscale("log")

        ax.legend()
