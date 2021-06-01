control package
===============

Control Base Classes 
--------------------
 
The ControllerFactory Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.control.controller.ControllerFactory
   :members: __call__, get_configuration_space

The Controller Class
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.control.controller.Controller
   :members:


Supported Controllers
---------------------

Linear Quadratic Regulator (LQR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.control.LQRFactory

Iterative Linear Quadratic Regulator (iLQR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.control.IterativeLQRFactory

Direct Transcription (DT)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.control.DirectTranscriptionControllerFactory

Model Predictive Path Integral (MPPI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.control.MPPIFactory

Zero Controller
^^^^^^^^^^^^^^^
.. autoclass:: autompc.control.ZeroControllerFactory
