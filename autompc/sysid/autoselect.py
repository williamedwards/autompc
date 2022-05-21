from typing import List,Optional
import ConfigSpace as CS

from ..tunable import TunablePipeline
from ..system import System
from .model import Model
from .mlp import MLP
from .arx import ARX
from .koopman import Koopman
from .sindy import SINDy
from .largegp import ApproximateGPModel


class AutoSelectModel (Model, TunablePipeline):
    """A Model that selects from a set of models.  By default, adds all known
    models.
    """
    def __init__(self, system : System, models : Optional[List[Model]]= None):
        Model.__init__(self,system,'AutoSelect')
        TunablePipeline.__init__(self)
        if models is None:
            models = [MLP(system),ARX(system),Koopman(system),SINDy(system),ApproximateGPModel(system)]
        TunablePipeline.add_component(self,'model',models)
        self.selected_model = None # type: Model
        self.train_time_limit = None
    
    def add_model(self, model : Model) -> None:
        """
        Add a model which is an option for tuning.
        Multiple model factories can be added and the tuner
        will select between them.

        Parameters
        ----------
        model : Model
            Model to be considered for tuning.
        """
        self.add_component_option('model',model)
    
    def get_default_config_space(self):
        return CS.ConfigurationSpace()
    
    def get_default_config(self):
        return None
    
    def get_config_space(self):
        return TunablePipeline.get_config_space(self)

    def set_config(self, cfg : CS.Configuration) -> None:
        if cfg is None: return
        TunablePipeline.set_config(self,cfg)
        pipeline = TunablePipeline.get_configured_pipeline(self)
        self.selected_model = pipeline[0]
        #configure budget properties
        self.selected_model.set_train_time(self.train_time_limit)
    
    def selected(self) -> Model:
        return self.selected_model
    
    @property
    def models(self) -> List[Model]:
        return self._components['model']

    def train(self, trajs):
        if self.selected_model is None:
            raise RuntimeError("Must set a valid model configuration before training")
        self.selected_model.train(trajs)

    def set_train_budget(self, seconds=None):
        self.train_time_limit = seconds
        if self.selected_model is not None:
            return self.selected_model.set_train_budget(seconds)

    def pred(self, state, ctrl):
        if self.selected_model is None:
            raise RuntimeError("Must set a valid model configuration before pred")
        return self.selected_model.pred(state,ctrl)
    
    def pred_diff(self, state, ctrl):
        if self.selected_model is None:
            raise RuntimeError("Must set a valid model configuration before pred_diff")
        return self.selected_model.pred_diff(state,ctrl)
    
    def pred_diff_batch(self, state, ctrl):
        if self.selected_model is None:
            raise RuntimeError("Must set a valid model configuration before pred_diff_batch")
        return self.selected_model.pred_diff_batch(state,ctrl)

    @property
    def state_dim(self):
        if self.selected_model is None:
            raise RuntimeError("Must set a valid model configuration before state_dim")
        return self.selected_model.state_dim
    
    def traj_to_state(self, traj):
        if self.selected_model is None:
            raise RuntimeError("Must set a valid model configuration before traj_to_state")
        return self.selected_model.traj_to_state(traj)
    
    def update_state(self, state, ctrl, new_obs):
        if self.selected_model is None:
            raise RuntimeError("Must set a valid model configuration before traj_to_state")
        return self.selected_model.update_state(state,ctrl,new_obs)

    def get_obs(self,state):
        if self.selected_model is None:
            raise RuntimeError("Must set a valid model configuration before get_obs")
        return self.selected_model.get_obs(state)

    def clear(self) -> None:
        if self.selected_model is None:
            return
        return self.selected_model.clear()

    def to_linear(self):
        if self.selected_model is None:
            raise RuntimeError("Must set a valid model configuration before to_linear")
        return self.selected_model.to_linear()

    def get_parameters(self):
        if self.selected_model is None:
            raise RuntimeError("Must set a valid model configuration before get_parameters")
        return self.selected_model.get_parameters()

    def set_parameters(self, params):
        if self.selected_model is None:
            raise RuntimeError("Must set a valid model configuration before set_parameters")
        return self.selected_model.set_parameters(params)

    @property
    def is_linear(self) -> bool:
        if self.selected_model is None:
            return False
        return self.selected_model.is_linear

    @property
    def is_diff(self) -> bool:
        if self.selected_model is None:
            return False
        return self.selected_model.is_diff
