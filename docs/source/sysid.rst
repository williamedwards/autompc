sysid package
=============

SysID Base Classes 
------------------
 
The ModelFactory Class
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.model.ModelFactory
   :members:

The Model Class
^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.model.Model
   :members:


Supported System ID Models
--------------------------

Multi-layer Perceptron
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.MLPFactory

Sparse Identification of Nonlinear Dynamics (SINDy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.SINDyFactory

Autoregression (ARX)
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.ARXFactory

Koopman
^^^^^^^

.. autoclass:: autompc.sysid.KoopmanFactory

Approximate Gaussian Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: autompc.sysid.ApproximateGPModelFactory
