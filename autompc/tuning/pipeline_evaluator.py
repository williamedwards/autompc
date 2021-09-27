from abc import ABC, abstractmethod

class PipelineEvaluator(ABC):
    def __init__(self, system, task, trajs, pipeline):
        self.system = system
        self.task = task
        self.trajs = trajs
        self.pipeline = pipeline

    @abstractmethod
    def __call__(self, cfg):
        """
        Evaluates configuration.  Returns uncertainy distribution over
        scores, represented as quantile function.
        """
        raise NotImplementedError