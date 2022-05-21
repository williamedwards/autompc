# Created by William Edwards

from pdb import set_trace

import numpy as np
import numpy.linalg as la

from ..sysid.metrics import get_model_rmse, get_model_rmsmens, get_model_abs_error

class KstepPredAccGraph:
    """
    Create k-step model prediction accuracy graph.
    """
    def __init__(self, system, trajs, kmax, logscale=False, metric="rmse"):
        """
        Parameters
        ----------
        system : System
            System on which models are being evaluted

        trajs : List of Trajectory
            Evaluation trajectory set

        kmax : int
            Maximum horizon to evaluate

        logscale : bool
            Use log scale on y-axis if true

        metric : string
            Prediction accuracy metric to use. One of "rmse" or "rmsmens"
        """
        self.kmax = kmax
        self.trajs = trajs
        self.logscale = logscale
        self.models = []
        self.labels = []
        self.plot_kwargs = []

        if metric == "rmse":
            self.metric = get_model_rmse
        elif metric == "rmsmens":
            self.metric = get_model_rmsmens
        elif metric == "abserror":
            self.metric = get_model_abs_error
        else:
            raise ValueError("Unknown error metric")

    def add_model(self, model, label, plot_kwargs=dict()):
        """
        Add a model for comparison

        Parameters
        ----------
        model : Model
            Model to compare

        label : string
            Label for model
        
        plot_kwargs : dict
            kwargs to pass to matplotlib plot
        """
        self.models.append(model)
        self.labels.append(label)
        self.plot_kwargs.append(plot_kwargs)

            
    def __call__(self, fig, ax):
        """
        Create graph.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure in which to create graph

        ax : matplotlib.axes.Axes
            Axes in which to create graph
        """
        for model, label, kwargs in zip(self.models, self.labels, self.plot_kwargs):
            rmses = [self.metric(model, self.trajs, horizon) 
                        for horizon in range(1, self.kmax)] 
            ax.plot(rmses, label=label, **kwargs)

        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel("Prediction Error")
        if self.logscale:
            ax.set_yscale("log")

        ax.legend()
