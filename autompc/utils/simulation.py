# Standard library library
import sys

# Internal library includes
from ..trajectory import DynamicTrajectory, Trajectory
from ..dynamics import Dynamics, LambdaDynamics
from ..policy import Policy
from ..sysid.model import Model

# External library includes
import numpy as np
from tqdm import tqdm
from time import time

def rollout(dynamics : Dynamics, traj : Trajectory, start : int = 0, horizon : int = None) -> Trajectory:
    """
    Rollout model predictions using the controls from traj.

    Parameters
    ----------
    dynamics : Dynamics
        Dynamics model used for prediction.  Note: see LambdaDynamics

    traj : Trajectory
        Trajectory used to compute initial model state and
        for controls.

    start : int
        For a start index t, the model will begin by predicting
        the observation at index t+1 based on the trajectory history.

    horizon : int
        Number of steps to rollout model.  If not given, the whole
        trajectory will be rolled out.
    
    Returns
    -------
    traj : Trajectory
        Predicted trajectory of length start+horizon+1.
        Identical to input trajectory for first start+1
        steps.
    """
    if callable(dynamics):
        dynamics = LambdaDynamics(traj.system,dynamics)
    if isinstance(dynamics,Model):
        if not dynamics.is_trained:
            raise ValueError("Trying to do rollout() on an untrained dynamics model?")
    if horizon is None:
        horizon = len(traj)-start
    model_state = dynamics.traj_to_state(traj[:start+1])
    new_obs = []
    for t in range(start, start+horizon):
        model_state2 = dynamics.pred(model_state, traj[t].ctrl)
        assert model_state is not model_state2,"prediction function should not modify state in-place"
        model_state = model_state2
        obs = dynamics.get_obs(model_state)
        new_obs.append(obs)
    return Trajectory(traj.system,np.concatenate((traj.obs[:start+1],new_obs[:-1]),0),traj.ctrls[:start+horizon+1])


def simulate(policy : Policy, init_obs, dynamics : Dynamics, term_cond=None, max_steps=10000, silent=False) -> Trajectory:
    """
    Simulate a policy with respect to a dynamics function.

    Note that if the policy is a controller, it is not reset at the start --
    this will need to be done manually.

    Parameters
    ----------
    policy : Policy
        Controller to simulate

    init_obs : numpy array of size controller.system.obs_dim
        Initial observation

    dynamics : Dynamics
        The dynamics function.

    term_cond : Function Trajectory -> bool
        Function which returns true when termination condition is met.

    max_steps : int
        Maximum number of simulation steps allowed.  Default is 10000.

    silent : bool
        Suppress output if True.
    """
    if callable(dynamics):
        dynamics = LambdaDynamics(policy.system,dynamics)
    if isinstance(dynamics,Model):
        if not dynamics.is_trained:
            raise ValueError("Trying to do simulate() on an untrained dynamics model?")
    x = init_obs
    traj = DynamicTrajectory(dynamics.system)
    
    simstate = dynamics.init_state(init_obs)
    if silent:
        itr = range(max_steps)
    else:
        itr = tqdm(range(max_steps), file=sys.stdout)
    for _  in itr:
        u = policy.step(x)
        traj.append(x,u)
        if term_cond is not None and term_cond(traj):
            break
        if len(traj) == max_steps:
            break

        simstate2 = dynamics.pred(simstate, u)
        assert simstate is not simstate2,"prediction function should not modify state in-place"
        simstate = simstate2
        x = dynamics.get_obs(simstate)
    #finalize Trajectory
    return traj.freeze()
