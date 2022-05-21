from abc import abstractmethod
import copy
import ConfigSpace as CS
from ..tunable import Tunable
from ..system import System
from ..costs import Cost
from .ocp import OCP

class OCPTransformer(Tunable):
    """An OCPTransformer takes an existing OCP and produces another OCP.
    These transformers can convert costs to simpler forms, apply
    regularization, or convert hard state/control constraints to soft costs.
    
    For example, if ocp_orig is incompatible with an optimizer, we'd apply
    a transformer as follows::

        optimizer = MyOptimizer(system)
        optimizer.set_model(model)
        print(optimizer.is_compatible(model,ocp_orig))  #prints False
        #optimizer.set_ocp(ocp_orig)                    #this line would cause the optimizer to barf
        transformer = MyOCPTranformer(system)
        ocp = transformer(ocp_orig)
        optimizer.set_ocp(ocp)
        optimizer.step(optimizer.get_state())

    Transformers can be composed by addition `xform1 + xform2` or composition
    `xform1 @ xform2`.
    """
    def __init__(self, system : System, name : str):
        self.system = system
        self.name = name
        self.is_trained = False
        Tunable.__init__(self)

    @abstractmethod
    def get_default_config_space(self) -> CS.ConfigurationSpace:
        raise NotImplementedError

    def train(self, trajs) -> None:
        """Trainable subclasses should implement this to train on a list of
        trajectories.  Should then set self.is_trained = True.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, ocp : OCP) -> OCP:
        raise NotImplementedError

    def ocp_requirements(self) -> dict:
        """Returns a set of ocp properties that must hold for this transformer
        to work.  For example `{'are_obs_bounded':True}` specifies that this
        transformer only works when observations are bounded.
        """
        raise NotImplementedError

    @property
    def trainable(self) -> bool:
        """
        Returns true for trainable models.
        """
        return not (self.train.__func__ is OCPTransformer.train)

    @abstractmethod
    def get_prototype(self, config : CS.Configuration, ocp : OCP):
        """
        Returns a prototype of the output OCP for compatibility checking.
        """
        raise NotImplementedError

    def __add__(self, other):
        from .sum_transformer import SumTransformer
        if isinstance(other, SumTransformer):
            return other.__radd__(self)
        else:
            return SumTransformer(self.system, [self, other])
    
    def __matmul__(self,other):
        from .sequence_transformer import SequenceTransformer
        if isinstance(other, SequenceTransformer):
            return other.__rmatmul__(self)
        else:
            return SequenceTransformer(self.system, [self, other])


class PrototypeCost:
    def __init__(self):
        pass

class PrototypeOCP:
    """
    PrototypeOCP represents only the compatibility properties of
    an OCP.  This is used for checking compatibility with as a little
    overhead as possible.
    """
    def __init__(self, ocp : OCP, cost : Cost=None):
        self.system = ocp.system
        self.are_obs_bounded = ocp.are_obs_bounded
        self.are_ctrl_bounded = ocp.are_ctrl_bounded
        if cost is None:
            cost = ocp.cost
        self.cost = PrototypeCost()
        self.cost.properties = cost.properties

    def get_cost(self):
        return self.cost


class IdentityTransformer(OCPTransformer):
    """
    Factory which preserves the input OCP.
    """
    def __init__(self, system):
        super().__init__(system, "Identity")

    def get_default_config_space(self):
        cs = CS.ConfigurationSpace()
        return cs

    def is_compatible(self, ocp):
        return True
        
    def get_prototype(self, config, ocp):
        return PrototypeOCP(ocp)

    def __call__(self, ocp):
        return copy.deepcopy(ocp)
