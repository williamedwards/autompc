import numpy as np
from typing import List
from scipy.stats import norm,rv_histogram
from .control_evaluator import ControlEvaluationTrial

class ControlPerformanceMetric:
    """A Callable that gathers trial results into a numerical value
    for tuning.  Default implementation just averages the cost.
    """
    def __call__(self,trials : List[ControlEvaluationTrial]) -> float:
        trial_costs = []
        for t in trials:
            if hasattr(t, "unwrap"):
                t = t.unwrap()
            trial_costs.append([t.cost])
        return np.mean(trial_costs)


class ConfidenceBoundPerformanceMetric(ControlPerformanceMetric):
    """A performance metric that uses a quantile of some statistical
    distribution, and also allows incorporating evaluation time and
    infeasibility.

    Overall performance  for a trial is::

        cost + eval_time_weight * E[control evaluation time] +
            infeasible_cost * I[trial is infeasible].
    
    Then, the `quantile` of the trial performance distributions is
    returned as the overall score.

    Default aggregator fits a NormalDistribution to the trial values.
    """
    def __init__(self,quantile=0.5,eval_time_weight=0.0,infeasible_cost=100.0,aggregator=None):
        if aggregator is None:
            #aggregator = NormalDistribution
            aggregator = EmpiricalDistribution
        self.aggregator = aggregator
        self.quantile = quantile
        self.eval_time_weight = eval_time_weight
        self.infeasible_cost = infeasible_cost
    
    def __call__(self,trials : List[ControlEvaluationTrial]) -> float:
        costs = []
        for t in trials:
            if hasattr(t, "unwrap"):
                t = t.unwrap()
            c = t.cost + self.eval_time_weight*t.eval_time/len(t.traj)
            if t.term_cond.endswith('infeasible'):
                c += self.infeasible_cost
            costs.append(c)
        return self.aggregator(costs)(self.quantile)


 
class ConstantDistribution:
    def __init__(self, values):
        self.val = np.mean(values)

    def __call__(self, quantile):
        return self.val

    def __str__(self):
        return "<Constant Distribution, Val={}>".format(self.val)

class NormalDistribution:
    def __init__(self, values):
        self.mu = np.mean(values)
        self.sigma = np.std(values)

    def __call__(self, quantile):
        if self.sigma > 0:
            return norm.ppf(quantile, loc=self.mu, scale=self.sigma)

    def __str__(self):
        return "<Normal Distribution, mean={}, std={}>".format(
                self.mu, self.sigma)

class HistogramDistribution:
    def __init__(self, values):
        self.dist = rv_histogram(np.histogram(values,len(values)))
        
    def __call__(self, quantile):
        return self.dist.ppf(quantile)

    def __str__(self):
        return "<Histogram Distribution, mean={}, std={}>".format(
                self.dist.mean(), self.dist.std())

class EmpiricalDistribution:
    def __init__(self, values):
        self.values = values 

    def __call__(self, quantile):
        return np.quantile(self.values, quantile)

    def __str__(self):
        return "<Empirical Distribution, mean={}, std={}>".format(
                np.mean(self.values), np.std(self.values))

