from abc import ABC,abstractmethod
import numpy as np
from .system import System
from .trajectory import Trajectory

class Policy(ABC):
    """
    An abstract base class for a policy, i.e., a function that generates a
    control for each new observation upon the step() call. 
    
    A policy may have internal state, and policies that do maintain internal
    state should implement the reset(), set_history(). get_state(), and
    set_state() methods.

    As an example, the OpenLoopPolicy executes a sequence of controls.
    """
    def __init__(self, system : System):
        super().__init__()
        self.system = system

    @abstractmethod
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
        return

    def reset(self) -> None:
        """
        Resets the policy's internal state.
        """
        return

    def set_history(self, history : Trajectory) -> None:
        """
        Provide a prior history of observations for modelling purposes.

        Parameters
        ----------
        history : Trajectory
            History of the system.
        """
        return

    def set_state(self, state : dict) -> None:
        """
        Sets the current policy internal state.
        """
        pass

    def get_state(self) -> dict:
        """
        Gets the current policy state.
        """
        return dict()



class OpenLoopPolicy(Policy):
    """
    An open-loop controller that executes each of the controls in `controls`
    one after the other.  The last control is executed forever after
    len(controls) steps have been taken.
    """
    def __init__(self, system : System, controls : np.ndarray):
        if len(controls) == 0:
            raise ValueError("Invalid controls list, must have at least one control")
        if controls.shape[1] != system.ctrl_dim:
            raise ValueError("Invalid controls dimensions, got {} and need {}".format(controls.shape[1],system.ctrl_dim))
        super().__init__(system)
        self._controls = controls
        self._time = 0

    def step(self, obs):
        if self._time < len(self._controls):
            return self._controls[self._time]
        return self._controls[-1]

    def reset(self):
        self._time = 0

    def set_history(self, history : Trajectory):
        self._time = len(history)

    def set_state(self, state : dict) -> None:
        self._time = state['time']

    def get_state(self) -> dict:
        return {'time':self._time}


class LinearPolicy(Policy):
    """
    A closed loop linear policy K*(x-target) - k.
    """
    def __init__(self, system : System, gain : np.ndarray, offset : np.ndarray=None, target : np.ndarray=None):
        if gain.shape != (system.ctrl_dim,system.obs_dim):
            raise ValueError("Invalid gain dimensions, got {} and need {}".format(gain.shape,(system.ctrl_dim,system.obs_dim)))
        super().__init__(system)
        self._gain = gain
        if offset is None:
            self._offset = np.zeros(system.ctrl_dim)
        else:
            if offset.shape != (system.ctrl_dim,):
                raise ValueError("Invalid offset dimensions, got {} and need {}".format(offset.shape,(system.ctrl_dim,)))
            self._offset = offset
        if target is not None:
            if target.shape != (system.obs_dim,):
                raise ValueError("Invalid target dimensions, got {} and need {}".format(target.shape,(system.obs_dim,)))
            self._offset = self._offset - self._gain @ target

    def step(self, obs):
        return self._gain @ obs + self._offset


class ClippedPolicy(Policy):
    """A policy that clips the output of another policy to control bounds.
    """
    def __init__(self, policy : Policy, ctrl_bounds : np.ndarray):
        if ctrl_bounds.shape != (policy.system.ctrl_dim,2):
            raise ValueError("Invalid control bounds shape")
        super().__init__(policy.system)
        self._policy = policy
        self._ctrl_bounds = ctrl_bounds
    
    def step(self,obs):
        u = self._policy.step(obs)
        return np.clip(u, self._ctrl_bounds[:,0], self._ctrl_bounds[:,1])
    
    def reset(self):
        self._policy.reset()

    def set_history(self, history : Trajectory):
        self._policy.set_history(history)

    def set_state(self, state : dict) -> None:
        self._policy.set_state(state)

    def get_state(self) -> dict:
        return self._policy.get_state()
    