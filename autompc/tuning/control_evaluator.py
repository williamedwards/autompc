from abc import ABC, abstractmethod
from scipy.stats import norm

class ControlEvaluator(ABC):
    def __init__(self, system, task, trajs):
        self.system = system
        self.task = task
        self.trajs = trajs

    @abstractmethod
    def __call__(self, controller):
        """
        Evaluates configuration.  Returns uncertainy distribution over
        scores, represented as quantile function.
        """
        raise NotImplementedError

class ConstantDistribution:
    def __init__(self, val):
        self.val = val

    def __call__(self, quantile):
        return self.val

    def __str__(self):
        return "<Constant Distribution, Val={}>".format(self.val)

class NormalDistribution:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, quantile):
        return norm.ppf(quantile, loc=self.mu, scale=self.sigma)

    def __str__(self):
        return "<Normal Distribution, mean={}, std={}>".format(
                self.mu, self.sigma)
