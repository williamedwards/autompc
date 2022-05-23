from abc import ABC,abstractmethod
from typing import Tuple,Callable
import numpy as np

from .trajectory import Trajectory
from .system import System

class Dynamics(ABC):
    """An abstract base class for some dynamics model.  In general a dynamics
    model can have different state dimensions than observation dimensions.
    To specify a fully observable system, use the FullyObservableDynamics
    class.

    To simulate a model given an initial observation(s) and controls, the
    methods in `autompc.utils.simulation` are convenient.  Or, you could
    manually call::

        x0 = init_state(obs0)
        x1_pred = model.pred(x0,u0)
        [execute u0]
        obs1 = [read new observation]
        x1 = model.update_state(x0,u0,obs1)
        x2_pred = model.pred(x1,u1)
        ...
    
    """
    def __init__(self, system : System):
        self.system = system
    

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """
        Returns the size of the system state
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def state_system(self) -> System:
        """
        Returns a system sized and named appropriately for the state space
        (rather than observation space)
        """
        raise NotImplementedError

    @abstractmethod
    def traj_to_state(self, traj : Trajectory):
        """
        Parameters
        ----------
            traj : Trajectory
                State and control history up to present time
        Returns
        -------
            state : numpy array of size self.state_dim
               Corresponding system state
        """
        raise NotImplementedError

    def init_state(self, obs : np.ndarray) -> np.ndarray:
        """
        Returns system state for an initial observation.

        Parameters
        ----------
            obs : numpy array of size self.system.obs_dim
                Initial observation.
        Returns
        -------
            state : numpy array of size self.state_dim
                Initial state.
        """
        traj = Trajectory.zeros(self.system, 1)
        traj[0].obs[:] = obs
        return self.traj_to_state(traj)

    @abstractmethod
    def update_state(self, state : np.ndarray, ctrl : np.ndarray, new_obs : np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            state : numpy array of size self.state_dim
                Current system state
            ctrl : numpy array of size self.system.ctrl_dim
                Control input executed from current state.
            new_obs : numpy array of size self.system.obs_dim
                New observation on time t+1
        Returns
        -------
            state : numpy array of size self.state_dim
                System state after control and observation
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_obs(self, state : np.ndarray) -> np.ndarray:
        """Extract the predicted observation from a predicted state variable.
        Parameters
        ----------
            state : numpy array of size self.state_dim
                System state
        Returns
        -------
            obs : numpy array of size self.system.ctrl_dim
                Estimated observation at state
        """
        raise NotImplementedError

    @abstractmethod
    def pred(self, state : np.ndarray, ctrl : np.ndarray) -> np.ndarray:
        """
        Run model prediction.

        Parameters
        ----------
            state : Numpy array of size self.state_dim
                System state at time t
            ctrl : Numpy array of size self.system.ctrl_dim
                Control applied at time t
        Returns
        -------
            state : Numpy array of size self.state_dim
                Predicted system state at time t+1
        """
        raise NotImplementedError

    def pred_batch(self, states : np.ndarray, ctrls : np.ndarray) -> np.ndarray:
        """
        Run batch model predictions.  Depending on the model, this can
        be much faster than repeatedly calling pred.

        Parameters
        ----------
            state : Numpy array of size (N, self.state_dim)
                N system input states
            ctrl : Numpy array of size (N, self.system.ctrl_dim)
                N controls
        Returns
        -------
            state : Numpy array of size (N, self.state_dim)
                N predicted states
        """
        n = self.state_dim
        m = states.shape[0]
        out = np.empty((m, n))
        for i in range(m):
            out[i,:] = self.pred(states[i,:], ctrls[i,:])
        return out

    def pred_diff(self, state : np.ndarray, ctrl : np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Run model prediction and compute gradients.

        Parameters
        ----------
            state : Numpy array of size self.state_dim
                State at time t
            ctrl : Numpy array of size self.system.ctrl_dim
                Control at time t
        Returns
        -------
            state : Numpy array of size self.state_dim
                Predicted state at time t+1
            state_jac : Numpy  array of shape (self.state_dim, 
                        self.state_dim)
                Gradient of predicted state wrt to state
            ctrl_jac : Numpy  array of shape (self.state_dim, 
                       self.ctrl_dim)
                Gradient of predicted state wrt to ctrl
        """
        raise NotImplementedError

    def pred_diff_batch(self, states : np.ndarray, ctrls : np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """
        Run model prediction and compute gradients in batch.

        Parameters
        ----------
            state : Numpy array of shape (N, self.state_dim)
                N input system states
            ctrl : Numpy array of size (N, self.system.ctrl_dim)
                N input controls
        Returns
        -------
            state : Numpy array of size (N, self.state_dim)
                N predicted states
            state_jac : Numpy  array of shape (N, self.state_dim, 
                        self.state_dim)
                Gradient of predicted states wrt to state
            ctrl_jac : Numpy  array of shape (N, self.state_dim, 
                       self.ctrl_dim)
                Gradient of predicted states wrt to ctrl
        """
        n = self.state_dim
        m = states.shape[0]
        out = np.empty((m, n))
        state_jacs = np.empty((m, n, n))
        ctrl_jacs = np.empty((m, n, self.system.ctrl_dim))
        for i in range(m):
            out[i,:], state_jacs[i,:,:], ctrl_jacs[i,:,:] = \
                self.pred_diff(states[i,:], ctrls[i,:])
        return out, state_jacs, ctrl_jacs
    

class FullyObservableDynamics(Dynamics):
    """An abstract base class for a fully-observable dynamics model.
    Subclass still needs to implement pred() and optionally
    pred_diff()/pred_diff_batch().
    """
    @property
    def state_dim(self) -> int:
        return self.system.obs_dim
    
    @property
    def state_system(self) -> System:
        return self.system

    def traj_to_state(self, traj : Trajectory):
        return np.copy(traj.obs[-1])

    def init_state(self, obs : np.ndarray) -> np.ndarray:
        return np.copy(obs)
    
    def get_obs(self, state):
        return state

    def update_state(self, state : np.ndarray, new_ctrl : np.ndarray, new_obs : np.ndarray) -> np.ndarray:
        return np.copy(new_obs)
    

class LambdaDynamics(FullyObservableDynamics):
    """A helper class x[t+1] = f(x[t],u[t]) for converting Python functions
    into a Dynamics class.
    """
    def __init__(self, system: System, f : Callable):
        super().__init__(system)
        self._f = f

    def pred(self, state, ctrl):
        return self._f(state,ctrl)


class LinearDynamics(FullyObservableDynamics):
    """A standard linear time invariant system model::

        x[t+1] = A*x[t] + B*u[t]
        
    An optional drift term c can also be included, in which case::

        x[t+1] = A*x[t] + B*u[t] + c
    
    """
    def __init__(self, system: System, A : np.ndarray, B : np.ndarray, c : np.ndarray = None):
        super().__init__(system)
        if A.shape != (system.obs_dim,system.obs_dim):
            raise ValueError("Invalid shape for A matrix, got {}, need {}".format(A.shape,(system.obs_dim,system.obs_dim)))
        if B.shape != (system.obs_dim,system.ctrl_dim):
            raise ValueError("Invalid shape for B matrix, got {}, need {}".format(B.shape,(system.obs_dim,system.ctrl_dim)))
        self.A, self.B = A,B
        if c is None:
            self.c = np.zeros(system.obs_dim)
        else:
            if c.shape != (system.obs_dim,):
                raise ValueError("Invalid shape for c vector, got {}, need {}".format(c.shape,(system.obs_dim,)))
            self.c = c

    def pred(self, state, ctrl):
        return self.A @ state + self.B @ ctrl + self.c

    def pred_diff(self, state, ctrl):
        xpred = self.A @ state + self.B @ ctrl + self.c
        return xpred, self.A, self.B

    def pred_diff_batch(self, states, ctrls):
        xpreds = states @ self.A.T + ctrls @ self.B.T + self.c
        N = len(self.states)
        return xpreds,np.stack([self.A]*N).np.stack([self.B]*N)


class LinearizedDynamics(Dynamics):
    """A helper class that linearizes a nonlinear dynamics model about a
    reference state x0."""
    def __init__(self, nonlinear_model : Dynamics, x0):
        super().__init__(nonlinear_model.system)
        self.x0 = x0
        self._model = nonlinear_model
        const, self.A, self.B = nonlinear_model.pred_diff(x0,  np.zeros(self.system.ctrl_dim))
        #x[t+1] ~= const + A*(x-x0) + Bu = Ax + Bu + const - A*x0
        self.c = const - self.A @ x0

    @property
    def state_dim(self):
        return self._model.state_dim
    
    @property
    def state_system(self):
        return self._model.state_system

    def get_obs(self, state):
        return self._model.get_obs(state)

    def traj_to_state(self, traj):
        return self._model.traj_to_state(traj)

    def update_state(self, state, new_ctrl, new_obs):
        return self._model.update_state(state,new_ctrl,new_obs)

    def to_linear(self):
        return np.copy(self.A), np.copy(self.B), self.c

    def pred(self, state, ctrl):
        return self.A @ state + self.B @ ctrl + self.c

    def pred_diff(self, state, ctrl):
        xpred = self.A @ state + self.B @ ctrl + self.c
        return xpred, self.A, self.B
    
    def pred_diff_batch(self, states, ctrls):
        xpreds = states @ self.A.T + ctrls @ self.B.T + self.c
        N = len(self.states)
        return xpreds,np.stack([self.A]*N).np.stack([self.B]*N)
