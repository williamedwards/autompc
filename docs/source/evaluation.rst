evaluation package
==================

Base Classes
^^^^^^^^^^^^

Model Evaluator
---------------

.. autoclass:: autompc.evaluation.ModelEvaluator
   :members: __init__, __call__

Evaluator Types
^^^^^^^^^^^^^^^

HoldoutModelEvaluator
---------------------

.. autoclass:: autompc.evaluation.HoldoutModelEvaluator
   :members: __init__

Model Metrics
^^^^^^^^^^^^^

.. autofunction:: autompc.evaluation.model_metrics.get_model_rmse

.. autofunction:: autompc.evaluation.model_metrics.get_model_rmsmens
