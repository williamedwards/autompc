# Created by William Edwards (wre2@illinois.edu)
# Refactoring by Kris Hauser (kkhauser@illinois.edu)

from abc import ABC, abstractmethod
from functools import partial
import numpy as np
from typing import Callable,List,Union
from ConfigSpace import Configuration
from ..sysid.metrics import get_model_rmse,get_model_rmsmens
from ..trajectory import Trajectory
from ..sysid.model import Model
from .parallel_utils import SerialBackend

class ModelEvaluator(ABC):
    """
    The ModelEvaluator class evaluates models by prediction accuracy.
    """
    def __init__(self, trajs : List[Trajectory], metric : Union[str,Callable], quantile=None, horizon:int=1):
        """s
        Parameters
        ----------
        trajs : List of Trajectory
            Trajectories to be used for evaluation
        metric : string or function (model, List[Trajectory], horizon) -> float)
            Metric which evaluates the model against a set of trajectories.
            If string, one of "rmse", "rmsmens". See `model_metrics` for
            more details.
        quantile : float or None
            If given, uses a quantile-based metric (i.e., optimize worst case
            performance with quantile=1)
        horizon : int
            Prediction horizon used in certain metrics. Default is 1.
        """
        self.trajs = trajs
        if isinstance(metric, str):
            if quantile is not None:
                raise NotImplementedError("TODO: quantile-based metrics")
            if metric == "rmse":
                self.metric = partial(get_model_rmse, horizon=horizon)
            elif metric == "rmsmens":
                self.metric = partial(get_model_rmsmens, horizon=horizon)
        else:
            self.metric = metric

    @abstractmethod
    def __call__(self, model : Model) -> float:
        """
        Reports how well the model configuration performs.  Lower
        is better.

        Parameters
        ----------
        model : Model
            Configured, untrained Model to evaluate.
            
        Returns
        -------
        score : float
            Evaluated score.
        """
        raise NotImplementedError

    def set_data_store(self, data_store):
        self._data_store = data_store



class HoldoutModelEvaluator(ModelEvaluator):
    """
    Evaluate model prediction accuracy according to a holdout set.
    """
    def __init__(self, *args, rng = None, holdout_prop = 0.1, holdout_set=None, verbose=False, **kwargs):
        """
        Parameters
        ----------
        trajs : List of Trajectory
            Trajectories to be used for evaluation
        metric : string or function (model, [Trajectory] -> float)
            Metric which evaluates the model against a set of trajectories.
            If string, one of "rmse", "rmsmens". See `model_metrics` for
            more details.
        horizon : int
            Prediction horizon used in certain metrics. Default is 1.
        rng : np.random.Generator
            Random number generator used to select  
        holdout_prop : float
            Proportion of dataset to hold out for evaluation
        holdout_set : Optional List of Trajectory
            This argument can be passed to explicitly set the holdout set, 
            rather than randomly selecting it.
        verbose : bool
            Whether to print information during evaluation
        """
        if rng is None:
            rng = np.random.default_rng()
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        if holdout_set is None:
            holdout_size = round(holdout_prop * len(self.trajs))
            holdout_indices = rng.choice(np.arange(len(self.trajs)), 
                    holdout_size, replace=False)
            self.holdout = [self.trajs[i] for i in sorted(holdout_indices)]
        else:
            self.holdout = holdout_set
        self.training_set = []
        for traj in self.trajs:
            if traj not in self.holdout:
                self.training_set.append(traj)

    def set_data_store(self, data_store):
        super().set_data_store(data_store)
        self.holdout = data_store.wrap(self.holdout)
        self.training_set = data_store.wrap(self.training_set)

    def __call__(self, model):
        m = model.clone()
        training_set = self.training_set
        holdout = self.holdout
        if hasattr(training_set, "unwrap"):
            training_set = training_set.unwrap()
        if hasattr(holdout, "unwrap"):
            holdout = holdout.unwrap()

        m.train(self.training_set)
        metric_value = self.metric(m, self.holdout)
        if self.verbose:
            print("Holdout score:",metric_value)
        return metric_value


class CrossValidationModelEvaluator(ModelEvaluator):
    """
    Evaluate model prediction accuracy according to k-fold cross validation.
    """
    def __init__(self, *args, rng = None, num_folds = 3, verbose=False, parallel_backend=None, **kwargs):
        """
        Parameters
        ----------
        trajs : List of Trajectory
            Trajectories to be used for evaluation
        metric : string or function (model, [Trajectory] -> float)
            Metric which evaluates the model against a set of trajectories.
            If string, one of "rmse", "rmsmens". See `model_metrics` for
            more details.
        horizon : int
            Prediction horizon used in certain metrics. Default is 1.
        rng : np.random.Generator
            Random number generator used to select  
        num_folds : int
            Number of folds to evaluate.
        verbose : bool
            Whether to print information during evaluation
        parallel_backend : ParallelBackend
            Backend to use to compute folds in parallel.
        """
        if num_folds <= 1:
            raise ValueError("Invalid number of folds")
        if rng is None:
            rng = np.random.default_rng()
        super().__init__(*args, **kwargs)
        if len(self.trajs) < num_folds:
            raise ValueError("Need at least as many trajectories as folds")
        self.verbose = verbose
        shuffled_trajs = rng.permutation(list(range(len(self.trajs))))
        self.folds = []
        splits = [(len(self.trajs)*(f))//num_folds for f in range(num_folds+1)]
        for f in range(num_folds):
            train = np.hstack((shuffled_trajs[:splits[f]],shuffled_trajs[splits[f+1]:]))
            test = shuffled_trajs[splits[f]:splits[f+1]]
            self.folds.append(([self.trajs[i] for i in train],[self.trajs[i] for i in test]))

        if parallel_backend is None:
            parallel_backend = SerialBackend()
        self.parallel_backend = parallel_backend

    def set_data_store(self, data_store):
        super().set_data_store(data_store)
        self.folds = [data_store.wrap(fold) for fold in self.folds]

    def _run_fold(self, model, fold):
        if hasattr(fold, "unwrap"):
            fold = fold.unwrap()
        if hasattr(model, "unwrap"):
            model = model.unwrap()
        train, test = fold
        m = model.clone()
        m.train(train)
        metric_value = self.metric(m, test)
        return metric_value

    def __call__(self, model):
        values = self.parallel_backend.map(partial(self._run_fold, model), self.folds)
        if self.verbose:
            print("Cross-validation values:",values)
        return np.mean(values)