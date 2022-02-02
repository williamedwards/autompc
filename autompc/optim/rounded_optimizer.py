from .optimizer import Optimizer
import numpy as np

class RoundedOptimizer(Optimizer):
    def __init__(self, system, optimizer):
        self.optimizer = optimizer
        super().__init__(system, "Rounded" + optimizer.name)

    def get_default_config_space(self):
        return self.optimizer.get_default_config_space()

    def set_config(self, config):
        self.optimizer.set_config(config)
        
    def traj_to_state(self, *args, **kwargs):
        return self.controller.traj_to_state(*args, **kwargs)
    
    def run(self, *args, **kwargs):
        u = self.optimizer.run(*args, **kwargs)
        return np.around(u)

    def reset(self):
        self.optimizer.reset()

    def set_ocp(self, ocp):
        self.optimizer.set_ocp(ocp)

    def set_model(self, model):
        self.optimizer.set_model(model)

    def get_state(self):
        return self.optimizer.get_state()

    def set_state(self, state):
        self.optimizer.set_state(state)
    