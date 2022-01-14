
from .sysid.model import ModelFactory, Model
from .control.controller import Controller, ControllerFactory
from .ocp import OCP, OCPFactory

class ControllerFactory:
    """
    The ControllerFactory combines a model_factory, optimizer_factory,
    and ocp_factory and has joint configuration space over 
    its components.
    """
    def __init__(self, model_factory, optimizer_factory, ocp_factory):
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.ocp_factory = ocp_factory

    def create(self, cfg, trajs, ocp):
        model = self.model_factory(cfg, trajs)
        optimizer_factory_partial = self.optimizer_factory_partial(cfg)
        ocp_factory_partial = self.ocp_factory(cfg, trajs)

        return Controller(model, optimizer_factory_partial, ocp_factory_partial, ocp)

    def get_configuration_space(self):
        pass

class Controller:
    def __init__(self, model, optimizer_factory_partial, ocp_factory_partial, ocp):
        self.model = model
        self.optimizer_factory_parital = optimizer_factory_partial
        self.ocp_factory_partial = ocp_factory_partial
        self.ocp = ocp
        self.reset()

    def reset(self):
        # Instantiate optimizer and transformed ocp
        self.transformed_ocp = self.ocp_factory_partial(self.ocp)
        self.optimizer = self.optimizer_factory_partial(self.model, self.transformed_ocp)
        self.model_state = None
        self.last_control = None

    def set_history(self, history):
        """
        Provide a prior history of observations for modelling purposes.

        Parameters
        ----------
        history : Trajectory
            History of the system.
        """
        self.model_state = self.model.traj_to_state(history)

    def run(self, obs):
        # Returns control, handles model state updates
        if self.model_state is None:
            self.model_state = self.model.init_state(obs)
        elif not self.last_control is None:
            self.model_state = self.model.update_state(self.model_state, self.last_control, obs)
        
        control = optimizer.run(self.model_state)
        self.last_control = control

        return control

    def get_state(self):
        # Returns pickleable state object
        # combining model and optimizer state
        return {"model_state" : self.model_state,
            "last_control" : self.last_control,
            "optimizer_state" : self.optimizer.get_state()}

    def set_state(self, state):
        # Sets controller (model/optimizer) state
        self.model_state = state["model_state"]
        self.last_control = state["last_control"]
        self.optimizer.set_state(state["optimizer_state"])