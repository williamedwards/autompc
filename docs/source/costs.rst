costs package
=============

Cost Base Classes
^^^^^^^^^^^^^^^^^

The CostFactory Class
---------------------
.. autoclass:: autompc.costs.CostFactory
   :members: get_configuration_space, __call__

The Cost Class
--------------
.. autoclass:: autompc.costs.Cost
   :members: __call__, get_cost_matrices, get_goal, eval_obs_cost, eval_obs_cost_diff, eval_obs_cost_hess, eval_ctrl_cost, eval_ctrl_cost_diff, eval_ctrl_cost_hess, eval_term_obs_cost, eval_cost_cost_diff, eval_term_obs_cost_hess, is_quad, is_convex, is_diff, is_twice_diff


Cost Factory Classes
^^^^^^^^^^^^^^^^^^^^

QuadCostFactory
---------------
.. autoclass:: autompc.costs.QuadCostFactory

GaussRegFactory
---------------
.. autoclass:: autompc.costs.GaussRegFactory

SumCostFactory
--------------
.. autoclass:: autompc.costs.SumCostFactory

Cost Classes
^^^^^^^^^^^^

QuadCost
--------
.. autoclass:: autompc.costs.QuadCost
   :members: __init__

SumCost
------
.. autoclass:: autompc.costs.SumCost
   :members: __init__

ThresholdCost
-------------
.. autoclass:: autompc.costs.ThresholdCost
   :members: __init__

BoxThresholdCost
-------------
.. autoclass:: autompc.costs.BoxThresholdCost
   :members: __init__
