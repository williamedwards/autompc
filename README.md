Welcome to AutoMPC, a library for automating system identification and model predictive control.
AutoMPC can
 * Build SystemID models and Controllers
 * Evaluate and compare models and controllers
 * Tune controllers without requiring interactive access to the system
 * Provides a variety of controllers and optimizers

To see AutoMPC in action, check out this [example](https://htmlpreview.github.io/?https://github.com/williamedwards/autompc/blob/main/examples/0_MainDemo.html)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CNil-Cq24AjVtoArgWOW0ZvLNB-zcyGW).

## Why AutoMPC?

System ID and Model Predictive Control are powerful tools for building robot controllers, 
but getting them up and running can take a lot of engineering work.  Achieving good
performance typically requires careful selection of a number of hyperparameters,
including the MPC horizon, the terms of the objective function, and the parameters
of the System ID algorithm.  AutoMPC automates the selection of these hyperparameters
and provides a toolbox of algorithms to choose from.

## How does AutoMPC work?

AutoMPC tunes hyperparameters for the System ID, Control Optimizer, and objective function
using a dataset collected offline.  In other words, AutoMPC does not need to interact
with the robot during tuning.  This is accomplished by initially training a *surrogate*
dynamics model.  During tuning, the surrogate dynamics are then used to simulate candidate
controllers in order to evaluate closed-loop performance.

For more details, see our [paper](https://motion.cs.illinois.edu/papers/ICRA2021_Edwards_AutoMPC.pdf)

## How to use AutoMPC?

Check out our [main example](https://htmlpreview.github.io/?https://github.com/williamedwards/autompc/blob/main/examples/0_MainDemo.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CNil-Cq24AjVtoArgWOW0ZvLNB-zcyGW)
to see an overview of the AutoMPC workflow.

If you are interested, check out our [detailed examples](examples/readme.md) for more information on how to use the different parts of AutoMPC.

## What algorithms does AutoMPC support?

For System ID, AutoMPC supports
 * [Multi-layer Perceptrons](https://autompc.readthedocs.io/en/latest/source/sysid.html#multi-layer-perceptron)
 * [Sparse Identification of Nonlinear Dynamics (SINDy)](https://autompc.readthedocs.io/en/latest/source/sysid.html#sparse-identification-of-nonlinear-dynamics-sindy)
 * [Autoregression](https://autompc.readthedocs.io/en/latest/source/sysid.html#autoregression-arx)
 * [Koopman Operators](https://autompc.readthedocs.io/en/latest/source/sysid.html#koopman)
 * [Approximate Gaussian Processes](https://autompc.readthedocs.io/en/latest/source/sysid.html#approximate-gaussian-process)

For control optimization, AutoMPC supports
 * [Linear Quadratic Regulator](https://autompc.readthedocs.io/en/latest/source/control.html#linear-quadratic-regulator-lqr)
 * [Iterative LQR](https://autompc.readthedocs.io/en/latest/source/control.html#iterative-linear-quadratic-regulator-ilqr)
 * [Direct Transcription](https://autompc.readthedocs.io/en/latest/source/control.html#direct-transcription-dt)
 * [Model Path Predictive Integral](https://autompc.readthedocs.io/en/latest/source/control.html#model-predictive-path-integral-mppi)

AutoMPC is also extensible, so you can use our tuning process with your own System ID and control methods.  We'd also welcome contributions
of new algorithms to the package.

## Installation

 1. Clone the repository
 2. Install [PyTorch](https://pytorch.org/get-started/locally/)
 3. (Optional) For certain benchmarks to work, install OpenAI [gym](https://gym.openai.com/) and [Mujoco](http://www.mujoco.org/)
 4. (Optional) To use DirectTranscriptionController, install IPOPT solver and cyipopt binding. See [instructions](https://cyipopt.readthedocs.io/en/latest/install.html)
 5. Run `pip install -r requirements.txt`
 6. Run `pip install -e .`

## Documentation
[Python API Reference](https://autompc.readthedocs.io).

The documentation can also be built offline. This requires Sphinx to be installed,
which can be done by running
```
pip install sphinx
```

To build or re-build the documentation, run the following command from the `docs/` subdirectory.
```
make html
```

The documentation will be produced in `docs/html`.
