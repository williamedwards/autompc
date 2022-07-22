from .optimizer import Optimizer
import numpy as np

class RoundedOptimizer(Optimizer):
    def __init__(self, system, optimizer):
        self.optimizer = optimizer
        super().__init__(system, "Rounded" + optimizer.name)

    def is_compatible(self, *args, **kwargs):
        return self.optimizer.is_compatible(*args, **kwargs)

    def get_default_config_space(self):
        return self.optimizer.get_default_config_space()

    def set_config(self, config):
        self.optimizer.set_config(config)
        
    def step(self, *args, **kwargs):
        u = self.optimizer.step(*args, **kwargs)
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

    def get_traj(self):
        traj = self.optimizer.get_traj()
        traj.ctrls=np.around(traj.ctrls)
        return traj
    