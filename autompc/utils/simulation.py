# Standard library library
import sys

# Internal library includes
from ..trajectory import zeros, extend

# External library includes
import numpy as np
from tqdm import tqdm

def rollout(model, traj, start, horizon):
    """
    Rollout model predictions using the controls from traj.

    Parameters
    ----------
    model : Model
        Model used for prediction.

    traj : Trajectory
        Trajectory used to compute initial model state and
        for controls.

    start : int
        For a start index t, the model will begin by predicting
        the observation at index t+1 based on the trajectory history.

    horizon : int
        Number of steps to rollout model.

    Returns
    -------
    traj : Trajectory
        Predicted trajectory of length start+horizon+1.
        Identical to input trajectory for first start+1
        steps.
    """
    model_state = model.traj_to_state(traj[:start+1])
    out_traj = traj[:start+1]
    for t in range(start, start+horizon):
        model_state = model.pred(model_state, traj[t].ctrl)
        out_traj = extend(out_traj, [model_state[:model.system.obs_dim]], [traj[t].ctrl])
    return out_traj

def simulate(controller, init_obs, term_cond=None, dynamics=None, sim_model=None, max_steps=10000, ctrl_bounds=None, silent=False):
    """
    Simulate a controller with respect to a dynamics function or simulation model.

    Parameters
    ----------
    controller : Controller
        Controller to simulate

    init_obs : numpy array of size controller.system.obs_dim
        Initial observation

    term_cond : Function Trajectory -> bool
        Function which returns true when termination condition is met.

    dynamics : Function obs, control -> newobs
        Function defining system dynamics

    sim_model : Model
        Simulation model.  Used when dynamics is None

    max_steps : int
        Maximum number of simulation steps allowed.  Default is 10000.

    silent : bool
        Suppress output if True.
    """
    if dynamics is None and sim_model is None:
        raise ValueError("Must specify dynamics function or simulation model")

    sim_traj = zeros(controller.system, 1)
    x = np.copy(init_obs)
    sim_traj[0].obs[:] = x
    
    if dynamics is None:
        simstate = sim_model.traj_to_state(sim_traj)
    if silent:
        itr = range(max_steps)
    else:
        itr = tqdm(range(max_steps), file=sys.stdout)
    for _  in itr:
        u = controller.run(sim_traj[-1].obs)
        if ctrl_bounds is not None:
            u = np.clip(u, ctrl_bounds[:,0], ctrl_bounds[:,1])
        if dynamics is None:
            simstate = sim_model.pred(simstate, u)
            x = simstate[:controller.system.obs_dim]
        else:
            x = dynamics(x, u)
        sim_traj[-1].ctrl[:] = u
        sim_traj = extend(sim_traj, [x], 
                np.zeros((1, controller.system.ctrl_dim)))
        if term_cond is not None and term_cond(sim_traj):
            break
    return sim_traj
