import numpy as np
import matplotlib.pyplot as plt

from ..tuning.control_tuner import ControlTunerResult
from ..tuning.model_tuner import ModelTuneResult

def plot_tuning_curve(tune_result, ax=None):
    """
    Graph tuning curve for either controller or model tuning
    result.

    Parameters
    ----------
    tune_result : ModelTuneResult or ControlTuneResult
        Tuning result to plot
    ax : matplotlib.axes.Axes
        Axes object on which to create graph
    """
    if ax is None:
        ax = plt.gca()
    if isinstance(tune_result, ControlTunerResult):
        if tune_result.inc_truedyn_costs is not None:
            ax.plot(tune_result.inc_truedyn_costs, label="True Dyn. Cost") 
        ax.plot(tune_result.inc_costs, label="Surr. Cost")
        ax.set_xlabel("Tuning Iteration")
        ax.set_ylabel("Cost")
        ax.legend()
    elif isinstance(tune_result, ModelTuneResult):
        ax.plot(tune_result.inc_costs, label="Surr. Cost")
        ax.set_xlabel("Tuning Iteration")
        ax.set_ylabel("Model Error")

def plot_tuning_correlations(tune_result, option, style='auto', ax=None):
    """
    Graph scatter plot (numeric) or column plot (categorical) of option
    vs cost for either controller or model tuning result.

    Parameters
    ----------
    tune_result : ModelTuneResult or ControlTuneResult
        Tuning result to plot
    option : str
        A configuration option to correlate with the cost.
    ax : matplotlib.axes.Axes
        Axes object on which to create graph
    """
    if ax is None:
        ax = plt.gca()

    x,y = [],[]
    for cfg,cost in zip(tune_result.cfgs,tune_result.costs):
        if option in cfg:
            x.append(cfg[option])
            y.append(cost)
    if len(x)==0:
        raise ValueError("Invalid option {} specified, not a valid configuration option".format(option))
    try:
        x = [float(v) for v in x]
        if style == 'auto':
            style = 'scatter'
    except ValueError:
        if style == 'auto':
            style = 'column'

    ax.set_ylabel("Cost")
    ax.set_xlabel(option)
    if isinstance(tune_result, ControlTunerResult):
        if style == 'scatter':
            ax.scatter(x,y,label="Surr. dynamics")
        else:
            labels = sorted(list(set(x)))
            values = dict((i,[]) for i in labels)
            for a,b in zip(x,y):
                values[a].append(b)
            values = [values[i] for i in labels]
            means = [np.mean(v) for v in values]
            stds = [np.std(v) for v in values]
            ax.bar(labels,means,yerr=stds,capsize=7,label="Surr. dynamics")
        if tune_result.inc_truedyn_costs is not None:
            ytrue = []
            for cfg,cost in zip(tune_result.cfgs,tune_result.costs):
                if option in cfg:
                    ytrue.append(cost)
            if style == 'scatter':
                ax.scatter(x,ytrue,label="True dynamics")
            else:
                labels = sorted(list(set(x)))
                values = dict((i,[]) for i in labels)
                for a,b in zip(x,ytrue):
                    values[a].append(b)
                values = [values[i] for i in labels]
                means = [np.mean(v) for v in values]
                stds = [np.std(v) for v in values]
                ax.bar(labels,means,yerr=stds,capsize=7,label="True dynamics")
            ax.legend()
        
    elif isinstance(tune_result, ModelTuneResult):    
        if style == 'scatter':
            ax.scatter(x,y)
        else:
            labels = sorted(list(set(x)))
            values = dict((i,[]) for i in labels)
            for a,b in zip(x,y):
                values[a].append(b)
            values = [values[i] for i in labels]
            means = [np.mean(v) for v in values]
            stds = [np.std(v) for v in values]
            ax.bar(labels,means,yerr=stds,capsize=7,label="Surr. dynamics")
