from autompc.control import Controller
import numpy as np

class RoundedController(Controller):
    def __init__(self, system, task, model, controller):
        super().__init__(system, task, model)
        self.controller = controller
        
    def traj_to_state(self, *args, **kwargs):
        return self.controller.traj_to_state(*args, **kwargs)
    
    def run(self, *args, **kwargs):
        u, constate = self.controller.run(*args, **kwargs)
        return np.around(u), constate
    
    def is_compatible(self, *args, **kwargs):
        return self.controller.is_compatible(*args, **kwargs)
    
    @property
    def state_dim(self):
        return self.controller.state_dim