costs package
=============

Cost Base Classes
^^^^^^^^^^^^^^^^^

The Cost Class
--------------
.. autoclass:: autompc.costs.Cost
   :members: __call__, incremental, incremental_diff, incremental_hess, terminal, terminal_diff, terminal_hess, goal


Cost Subclasses
^^^^^^^^^^^^^^^

QuadCost
---------------
.. autoclass:: autompc.costs.QuadCost
   :members: __init__, get_cost_matrices

ThresholdCost
-------------
.. autoclass:: autompc.costs.ThresholdCost
   :members: __init__

BoxThresholdCost
----------------
.. autoclass:: autompc.costs.BoxThresholdCost
   :members: __init__
