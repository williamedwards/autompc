core classes
============

System
^^^^^^
.. autoclass:: autompc.System
   :members: __init__, controls, observations, ctrl_dim, obs_dim

Trajectories
^^^^^^^^^^^^

Trajectory
----------
.. autoclass:: autompc.Trajectory
   :members: __init__, system, size, obs, ctrls

TimeStep
--------
.. autoclass:: autompc.trajectory.TimeStep

zeros
-----
.. autofunction:: autompc.zeros

empty
-----
.. autofunction:: autompc.empty

extend
------
.. autofunction:: autompc.extend

Pipeline
^^^^^^^^
.. autoclass:: autompc.Pipeline
   :members: __init__, get_configuration_space, __call__
