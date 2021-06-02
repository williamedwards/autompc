Welcome to AutoMPC, a library for automating system identification and model predictive control.
AutoMPC can
 * Build SystemID models and Controllers
 * Evaluate and compare models and controllers
 * Tune controllers without requiring access to the system
 * Provides a variety of controllers and optimizers

To see AutoMPC in action, check out this example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1w19fIzYi4r50XI1pW64xUqN_JBbbyK2L/view?usp=sharing).

## Installation

 1. Clone the repository
 2. Install [PyTorch](https://pytorch.org/get-started/locally/)
 3. (Optional) For certain benchmarks to work, install OpenAI [gym](https://gym.openai.com/) and [Mujoco](http://www.mujoco.org/)
 4. (Optional) To use DirectTranscriptionController, install IPOPT solver and cyipopt binding. See [instructions](https://cyipopt.readthedocs.io/en/latest/install.html)
 5. Run `pip install -r requirements.txt`
 6. Run `pip install -e .`

## Examples
[Examples](examples/readme.md)

## Documentation
Find the full API documentation online [here](https://autompc.readthedocs.io).

Building the documentation requies Sphinx to be installed.  This can be done by running
```
pip install sphinx
```

To build or re-build the documentation, run the following command from the `docs/` subdirectory.
```
make html
```

The documentation will be produced in `docs/html`.
