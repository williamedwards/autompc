Welcome to AutoMPC, a library for automating system identification and model predictive control.
AutoMPC can
 * Build SystemID models and Controllers
 * Evaluate and compare models and controllers
 * Tune controllers without requiring access to the system
 * Provides a variety of controllers and optimizers

To see AutoMPC in action, check out this [example](examples/0_MainDemo.ipynb).

## Installation

 1. Clone the repository
 2. Install [PyTorch](https://pytorch.org/get-started/locally/)
 3. (Optional) For certain benchmarks to work, install OpenAI [gym](https://gym.openai.com/) and [Mujoco](http://www.mujoco.org/)
 4. Run `pip install -r requirements.txt`
 5. Run `pip install -e .`

## Examples
[Examples](examples/readme.md)

## Documentation
Building the documentation requies Sphinx to be installed.  This can be done by running
```
pip install sphinx
```

To build or re-build the documentation, run the following command from the `docs/` subdirectory.
```
make html
```

The documentation will be produced in `docs/html`.
