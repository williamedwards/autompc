Core classes
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

TimeStep
--------
.. autoclass:: autompc.trajectory.TimeStep

Dynamics and Policies
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: autompc.Dynamics
   :members:

.. autoclass:: autompc.Policy
   :members:


Tasks
^^^^^^
.. autoclass:: autompc.Task
   :members:

Tunables
^^^^^^^^^
.. autoclass:: autompc.tunable.Tunable
   :members:

.. autoclass:: autompc.tunable.TunablePipeline
   :members:

Controller
^^^^^^^^^^
.. autoclass:: autompc.Controller
   :members:

.. autoclass:: autompc.AutoSelectController
   :members:
