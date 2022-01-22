from .arx import ARX, ARXFactory
from .koopman import Koopman, KoopmanFactory
from .sindy import SINDy, SINDyFactory
#from .gp import GaussianProcess
from .mlp import MLP, MLPFactory
try:
    from .largegp import ApproximateGPModel, ApproximateGPModelFactory
except ImportError:
    pass
#from .linearize import LinearizedModel
