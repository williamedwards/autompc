
# Standard libary includes
from configparser import NoOptionError
import copy
from typing import List,Optional

# External libary includes
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC


# Internal library includes
from .utils.cs_utils import *
from .utils.exceptions import OptionalDependencyException
from . import sysid
from . import optim
from . import ocp
from .ocp.ocp_transformer import OCPTransformer,IdentityTransformer
from .ocp.sum_transformer import SumTransformer
from .system import System
from .sysid.model import Model
from .optim.optimizer import Optimizer
from .ocp.ocp import OCP
from .trajectory import Trajectory
from .tunable import Tunable,NonTunable,TunablePipeline
from .dynamics import Dynamics
from .policy import Policy

class ControllerStateError(Exception):
    pass

class Controller(TunablePipeline,Policy):
    """
    Controller is the core class of AutoMPC, representing all components
    of a tunable, data-driven model predictive controller, including 
    system ID model, optimizer, and control problem transformer, as well as
    all associated hyperparameters.
    """
    def __init__(self, system : System):
        super().__init__()
        self.system = system
        self.models = []
        self.optimizers = []
        self.ocp_transformers = [IdentityTransformer(system)]
        self.model = None                #type: Model
        self.optimizer = None            #type: Optimizer
        self.ocp_transformer = None      #type: OCPTransformer
        self.ocp = None                  #type: OCP

        self.add_component("model",[])
        self.add_component("constraint_transformer",[])
        self.add_component("cost_transformer",[])
        self.add_component("regularizer",[])
        self.add_component("optimizer",[])

    def set_model(self, model : Model) -> None:
        """
        Set the controller to use a single system ID model.

        Parameters
        ----------
            model : Model
                System ID model to set.
        """
        if not isinstance(model,Model):
            if not isinstance(model,Dynamics):
                raise ValueError("set_model() must be called with a Model or Dynamics object")
            self.models = []
            self.model = model
        else:
            self.models = [model]
        self.set_component("model",self.models)
        
    def set_models(self, models : List[Model]) -> None:
        """
        Set the available system ID models.

        Parameters
        ----------
            models : List[Model]
                Set of System ID models which can be selected.
        """
        self.models = models[:]
        self.set_component("model",self.models)

    def add_model(self, model : Model) -> None:
        """
        Add an available system ID model to the existing list.

        Parameters
        ----------
            model : Model
                System ID model to be added.
        """
        self.models.append(model)
        self.add_component_option("model",model)

    def set_optimizer(self, optimizer : Optimizer) -> None:
        """
        Set the controller to use a single optimizer.

        Parameters
        ----------
            optimizer : Optimizer
                Optimizer to set.
        """
        self.optimizers = [optimizer]
        self.set_component("optimizer",self.optimizers)

    def set_optimizers(self, optimizers : List[Optimizer]) -> None:
        """
        Set the available optimizers.

        Parameters
        ----------
            optimizers : List[Optimizer]
                Set of optimizers which can be selected.
        """
        self.optimizers = optimizers[:]
        self.set_component("optimizer",self.optimizers)

    def add_optimizer(self, optimizer : Optimizer) -> None:
        """
        Add an available optimizer to the existing list.

        Parameters
        ----------
            optimizer : Optimizer
                Optimizer to be added.
        """
        self.optimizers.append(optimizer)
        self.add_component_option("optimizer",optimizer)

    def set_ocp_transformer(self, ocp_transformer : OCPTransformer) -> None:
        """
        Set the controller to use a single OCP transformer.

        Parameters
        ----------
            ocp_transformer : OCPTransformer
                OCPTransformer to set.
        """
        self.ocp_transformers = [ocp_transformer]

    def set_ocp_transformers(self, ocp_transformers : List[OCPTransformer]) -> None:
        """
        Set the available OCP transformers.

        Parameters
        ----------
            ocp_transformers : List[OCPTransformer]
                Set of OCP transformers which can be selected.
        """
        self.ocp_transformers = ocp_transformers

    def add_ocp_transformer(self, ocp_transformer):
        """
        Add an available OCP transformer to the existing list.

        Parameters
        ----------
            ocp_transformer : OCPTransformer
                OCP transformer to be added.
        """
        self.ocp_transformers.append(ocp_transformer)

    def set_ocp(self, ocp):
        """
        Set the OCP. If the controller is already running, this will
        also update the current optimizer's target
        """
        self.ocp = ocp

        if len(self.ocp_transformers) == 1:
            self.ocp_transformer = self.ocp_transformers[0]
        if self.ocp_transformer:
            self.transformed_ocp = self.ocp_transformer(self.ocp)
        else:
            self.transformed_ocp = self.ocp
        if self.optimizer:
            self.optimizer.set_ocp(self.transformed_ocp)

    def get_config_space(self):
        """
        Returns the joint controller configuration space.
        """
        
        if not self.models:
            raise ControllerStateError("Must add model before config space can be generated")
        if not self.optimizers:
            raise ControllerStateError("Must add optimizer before config space can be generated")
        if not self.ocp_transformers:
            raise ControllerStateError("Must add OCP transformer before config space can be generated")

        #transformer pipeline is a little more complex
        dummy = NonTunable()
        dummy.name = '_'
        cost_transformers = []
        constraint_transformers = [dummy]
        regularizers = [dummy]
        for transformer in self.ocp_transformers:
            if 'Bound' in transformer.name and transformer.name != 'DeleteBoundsTransformer':
                constraint_transformers.append(transformer)
            elif 'Reg' in transformer.name:
                regularizers.append(transformer)
            elif transformer.name != 'Identity':
                cost_transformers.append(transformer)
        cost_transformers.append(dummy)
        
        if len(cost_transformers) > 1:
            self.set_component("cost_transformer", cost_transformers)
        self.set_component("constraint_transformer", constraint_transformers)
        if len(regularizers) > 1:
            self.set_component("regularizer", regularizers)

        self._forbid_incompatible_configurations()

        cs = super().get_config_space()
        return cs
    
    def _forbid_incompatible_configurations(self):
        for opt in self.optimizers:
            try:
                reqs = opt.model_requirements()
                for prop,value in reqs.items():
                    for model in self.models:
                        if getattr(model,prop) != value:
                            print("Forbidding model",model.name,"to be used with",opt.name,"due to property",prop)
                            self.forbid_incompatible_options('model',model.name,'optimizer',opt.name)
            except NotImplementedError:
                pass
        if self.ocp is not None:
            print("Checking compatibility with OCP")
            cost = self.ocp.get_cost()
            if cost.is_quad:  #no need to transform costs into quadratic form
                print("Quad cost")
                print([xform.name for xform in self.ocp_transformers])
                if any(xform.name == 'QuadCostTransformer' for xform in self.ocp_transformers):
                    print("Forbidding QuadCostTransformer")
                    self.forbid_option("cost_transformer", 'QuadCostTransformer')
            if not self.ocp.are_obs_bounded:
                #Can safely ignore the bounds transformers
                self.fix_option("constraint_transformer", '_')
            
            cost_transformers = []
            constraint_transformers = []
            regularizers = []
            for transformer in self.ocp_transformers:
                if 'Bound' in transformer.name and transformer.name != 'DeleteBoundsTransformer':
                    constraint_transformers.append(transformer)
                elif 'Reg' in transformer.name:
                    regularizers.append(transformer)
                elif transformer.name != 'Identity':
                    cost_transformers.append(transformer)
            if self.ocp.are_obs_bounded:
                for xform in constraint_transformers:
                    if not xform.is_compatible(self.ocp):
                        self.forbid_option('constraint_transformer',xform.name)
            for xform in cost_transformers:
                if not xform.is_compatible(self.ocp):
                    self.forbid_option('cost_transformer',xform.name)
            for xform in regularizers:
                if not xform.is_compatible(self.ocp):
                    self.forbid_option('regularizer',xform.name)

            for opt in self.optimizers:
                try:
                    ocp_reqs = opt.ocp_requirements()
                    failed_requirements = [prop for prop,value in ocp_reqs.items() if getattr(self.ocp,prop) != value]
                    if failed_requirements:
                        #not compatible with raw OCP's constraints, need to include a transformer
                        print("Requiring bound transformer for ocp to be used with",opt.name,"due to property",failed_requirements[0])
                        self.forbid_incompatible_options('constraint_transformer','_','optimizer',opt.name)
                        #TODO: check if any other constraint transformers don't make the cut
                except NotImplementedError:
                    pass
                try:
                    cost_reqs = opt.cost_requirements()
                    failed_requirements = [prop for prop,value in cost_reqs.items() if getattr(cost,prop) != value]
                    if failed_requirements:
                        print("Requiring cost transformer for ocp to be used with",opt.name,"due to property",failed_requirements[0])
                        self.forbid_incompatible_options('cost_transformer','_','optimizer',opt.name)
                except NotImplementedError:
                    pass

    def set_config(self, config : Configuration) -> None:
        """
        Set the controller configuration.

        Parameters
        ----------
            config : Configuration
                Configuration to set.
        """
        super().set_config(config)
        pipeline = self.get_configured_pipeline()
        self.model = pipeline[0]
        self.optimizer = pipeline[-1]
        assert self.model is not None
        assert self.optimizer is not None
        sequence = []
        for i in range(1,4):
            if pipeline[i] is not None and pipeline[i].name != '_':
                assert isinstance(pipeline[i],OCPTransformer)
                sequence.append(pipeline[i])
        if len(sequence) > 1:
            self.ocp_transformer = SumTransformer(sequence[0].system,sequence)
        elif sequence:
            self.ocp_transformer = sequence[0]
        else:
            self.ocp_transformer = IdentityTransformer(self.system)

    def set_model_hyper_values(self, name=None, **kwargs) -> None:
        """
        Set model and model hyperparameters by keyword argument.

        Parameters
        ---------- 
            name : str
                Name of model to set
            **kwargs
                Model hyperparameter values
        """
        if len(self.models) == 0:
            raise ControllerStateError("Must add a model before setting hyper values")
        return self.set_hyper_values("model",name,**kwargs)

    def set_optimizer_hyper_values(self, name=None, **kwargs) -> None:
        """
        Set optimizer and optimizer hyperparameters by keyword argument.

        Parameters
        ---------- 
            name : str
                Name of optimizer to set
            **kwargs
                Optimizer hyperparameter values
        """
        if len(self.optimizers) == 0:
            raise ControllerStateError("Must add an optimizer before setting hyper values")
        return self.set_hyper_values("optimizer",name,**kwargs)

    def set_ocp_transformer_hyper_values(self, name=None, **kwargs) -> None:
        """
        Set OCP factory and hyperparameters by keyword argument.

        Parameters
        ---------- 
            name : str
                Name of OCP Factory to set
            **kwargs
                OCP Factory hyperparameter values
        """
        if len(self.ocp_transformers) == 0:
            raise ControllerStateError("Must add an ocp_transformer before setting hyper values")
        if name is None and len(self.ocp_transformers) == 1:
            name = self.ocp_transformers[0].name
        elif name is None:
            raise ValueError("Multiple ocp_transformers are present so name must be specified")
        matches = [xform for xform in self.ocp_transformers if xform.name==name]
        if not matches:
            raise ValueError("Unrecognized ocp_transformer name")
        raise NotImplementedError("TODO: override chosen transformer in the config")
        self.ocp_transformer = matches[0]
        self.ocp_transformer.set_hyper_values(**kwargs)

    def check_config(self, config : Configuration, ocp : Optional[OCP]=None) -> bool:
        """
        Check if config is compatible.

        Parameters
        ----------
            config : Configuration
                Configuration to check.
            ocp : OCP
                OCP to use for compatibility checking. If None,
                will use preset controller OCP.
        """
        if ocp is None:
            ocp = self.ocp
        old_config = self._config
        self.set_config(config)
        if self.ocp_transformer:
            if not self.ocp_transformer.is_compatible(ocp):
                if old_config is not None: self.set_config(old_config)
                return False
            try:
                ocp_ptype = self.ocp_transformer.get_prototype(self.ocp_transformer.get_config(), ocp)
            except NotImplementedError:
                ocp_ptype = self.ocp_transformer(ocp)
        else:
            ocp_ptype = ocp

        if not self.optimizer.is_compatible(self.model, ocp_ptype):
            if old_config is not None: self.set_config(old_config)
            return False
        if old_config is not None: self.set_config(old_config)
        return True

    def clear_model(self) -> None:
        """
        Clear learned model parameters.
        """
        if self.model is not None:
            self.model.clear()

    def clear(self) -> None:
        """
        Clears the trajectory training set, model parameters, and
        current config selection.
        """
        self.reset()
        self.clear_model()
        self.model, self.optimizer, self.ocp_transformer = None, None, None

    def clone(self) -> 'Controller':
        """
        Returns a deep copy of the controller.
        """
        return copy.deepcopy(self)

    def build(self, trajs : List[Trajectory] = None) -> None:
        """
        Builds the controller given its current configuration.  This includes
        training the model, constructing the OPC, and initializing the optimizer.
        """
        if not self.ocp:
            raise ControllerStateError("Must call set_ocp() before build()")
        if not self.model:
            if len(self.models)==1:
                self.model = self.models[0]
            else:
                raise ControllerStateError("set_config() must be called before build")
        if not self.optimizer:
            if len(self.optimizers)==1:
                self.optimizer = self.optimizers[0]
            else:
                raise ControllerStateError("set_config() must be called before build")
        self.optimizer.set_model(self.model)
        if hasattr(self.model,'trainable') and self.model.trainable:
            if trajs:
                self.model.clear()
                self.model.train(trajs)
            elif self.model.is_trained:
                pass
            else:
                raise ControllerStateError("Specified model requires learning from trajectories.")
        if not self.ocp_transformer:
            if len(self.ocp_transformers) == 1:
                self.ocp_transformer = self.ocp_transformers[0]
        if self.ocp_transformer and self.ocp_transformer.trainable:
            if trajs:
                self.ocp_transformer.train(trajs)
            else:
                raise ControllerStateError("Specified OCP transformer requires learning from trajectories.")
                
        if self.ocp_transformer:
            self.transformed_ocp = self.ocp_transformer(self.ocp)
        else:
            self.transformed_ocp = self.ocp
        self.optimizer.set_ocp(self.transformed_ocp)

        self.reset()
    
    def is_built(self) -> bool:
        """Returns true if the controller is built."""
        if self.model is None: return False
        if self.optimizer is None: return False
        if hasattr(self.model,'trainable') and self.model.trainable:
            if not self.model.is_trained:
                return False
        if self.ocp_transformer and self.ocp_transformer.trainable:
            if not self.ocp_transformer.is_trained:
                return False
        return True

    def reset(self) -> None:
        """
        Resets the the optimizer state and model history.
        """
        self.reset_optimizer()
        self.reset_history()

    def reset_history(self) -> None:
        """
        Resets the model history.  This prevents historical observations
        from influencing current model predictions.
        """
        self.model_state = None
        self.last_control = None

    def reset_optimizer(self) -> None:
        """
        Re-initialize the optimizer and regenerates the OCP.
        This clears any internal optimizer states such as a trajectory guess.
        """
        # Instantiate optimizer and transformed ocp
        if self.ocp is None:
            raise ControllerStateError("Need to provide OCP before resetting optimizer")
        if self.optimizer is None:
            raise ControllerStateError("Need to build OCP before resetting optimizer")
        if self.ocp_transformer:
            self.transformed_ocp = self.ocp_transformer(self.ocp)
        else:
            self.transformed_ocp = self.ocp
        self.optimizer.set_model(self.model)
        self.optimizer.set_ocp(self.transformed_ocp)
        self.optimizer.reset()

    def set_history(self, history : Trajectory) -> None:
        """
        Provide a prior history of observations for modelling purposes.

        Parameters
        ----------
        history : Trajectory
            History of the system.
        """
        self.model_state = self.model.traj_to_state(history)

    def step(self, obs : np.ndarray) -> np.ndarray:
        """
        Pass the controller a new observation and generate a new control.

        Parameters
        ----------
            obs : numpy array of size self.system.obs_dim
                New observation
        Returns
        -------
            control : numpy array of size self.system.ctrl_dim
                Generated control
        """
        # Returns control, handles model state updates
        if not self.is_built():
            raise ControllerStateError("Must call build() before run()")
        if self.model_state is None:
            self.model_state = self.model.init_state(obs)
        elif not self.last_control is None:
            self.model_state = self.model.update_state(self.model_state, 
                    self.last_control, obs)
        
        control = self.optimizer.step(self.model_state)
        self.last_control = control

        return control

    def get_state(self) -> dict:
        """
        Returns a representation of the controller state.
        """
        return {"model_state" : self.model_state,
            "last_control" : self.last_control,
            "optimizer_state" : self.optimizer.get_state()}
    
    def set_state(self,state) -> dict:
        """
        Sets the current controller state.
        """
        self.model_state = state["model_state"]
        self.last_control = state["last_control"]
        self.optimizer.set_state(state["optimizer_state"])

    def get_optimized_traj(self) -> Trajectory:
        """
        Returns the last optimized trajectory, if available.
        """
        return self.optimizer.get_traj()


class AutoSelectController(Controller):
    """
    A version of the controller which comes with a default selection
    of models, optimizer, and OCP factories.  Current list is

    **Models**
     - MLP
     - ApproximateGP
     - SINDy
     - ARX
     - Koopman
    
    **Optimizers**
     - IterativeLQR
     - MPPI
     - LQR
     - DirectTranscription

    **OCP Transformers**
     - QuadCostTransformer
     - KeepBoundsTransformer
     - DeleteBoundsTransformer
     - GaussRegTransformer

    Note that this list may change in future versions as new algorithms are added to AutoMPC.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_models()
        self._add_optimizers()
        self._add_ocp_transformers()
        
    def _add_if_available(self, add_func, module, name):
        if not hasattr(module, name):
            return
        try:
            add_func(getattr(module, name)(self.system))
        except OptionalDependencyException:
            return

    def _add_models(self):
        self._add_if_available(self.add_model, sysid, "MLP")
        self._add_if_available(self.add_model, sysid, "ApproximateGP")
        self._add_if_available(self.add_model, sysid, "SINDy")
        self._add_if_available(self.add_model, sysid, "ARX")
        self._add_if_available(self.add_model, sysid, "Koopman")

    def _add_optimizers(self):
        self._add_if_available(self.add_optimizer, optim, "IterativeLQR")
        self._add_if_available(self.add_optimizer, optim, "MPPI")
        self._add_if_available(self.add_optimizer, optim, "LQR")
        self._add_if_available(self.add_optimizer, optim, "DirectTranscription")

    def _add_ocp_transformers(self):
        self._add_if_available(self.add_ocp_transformer, ocp, "QuadCostTransformer")
        self._add_if_available(self.add_ocp_transformer, ocp, "KeepBoundsTransformer")
        self._add_if_available(self.add_ocp_transformer, ocp, "DeleteBoundsTransformer")
        self._add_if_available(self.add_ocp_transformer, ocp, "GaussRegTransformer")
    
        