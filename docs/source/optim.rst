optim package
===============

Optim Base Classes 
--------------------
 
The Optimizer Class
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.optim.optimizer.Optimizer
   :members:


Supported Optimizers
--------------------

Linear Quadratic Regulator (LQR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.control.LQRFactory

Iterative Linear Quadratic Regulator (iLQR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.optim.IterativeLQR

Direct Transcription (DT)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.control.DirectTranscriptionControllerFactory

Model Predictive Path Integral (MPPI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.control.MPPIFactory

Zero Controller
^^^^^^^^^^^^^^^
.. autoclass:: autompc.control.ZeroControllerFactory
