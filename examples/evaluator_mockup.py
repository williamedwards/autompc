import numpy as np
import autompc as ampc

pendulum = ampc.System(["ang", "angvel"], ["torque"])

# Generate pendulum trajectories
trajs = ...

from autompc.evaluation import CrossFoldEvaluator
from autompc.scoring import KstepScore

score = KstepScore(5)
print(score(trajs[0], trained_model))
# Prints at 5-step prediction error for a single trajectory

evaluator = CrossFoldEvaluator(system, score, folds=3)
print(evaluator(trajs, untrained_model))
# Performs model training
# Prints out overall training score

# Use evaluation to perform automatic hyperparameter tuning
from ampc.tuning import SMAC

tuner = SMAC()
model = tuner(model, evaluator)
