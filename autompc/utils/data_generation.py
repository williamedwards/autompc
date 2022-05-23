# Created by William Edwards (wre2@illinois.edu), 2021-01-09

# Standard library includes
import sys
from pdb import set_trace
from typing import List

# External library includes
import numpy as np
import autompc as ampc

# Project includes
from ..policy import OpenLoopPolicy, Policy
from ..dynamics import Dynamics
from ..system import System
from ..ocp import OCP
from ..trajectory import Trajectory
from .simulation import simulate

class WhiteNoisePolicy(Policy):
    """A policy that just samples controls at random."""
    def __init__(self, system : System, ctrl_bounds : np.ndarray, rng = None):
        if rng is None:
            rng = np.random.default_rng()
        super().__init__(system)
        self.umin, self.umax = ctrl_bounds.T
        self.rng = rng

    def step(self, obs):
        return self.rng.uniform(self.umin,self.umax)

class BrownianNoisePolicy(Policy):
    """A policy that evolves a random walk."""
    def __init__(self, system : System, ctrl_bounds : np.ndarray, rate = 1.0, rng = None):
        if rng is None:
            rng = np.random.default_rng()
        super().__init__(system)
        self.umin, self.umax = ctrl_bounds.T
        self.rate = rate 
        self.rng = rng
        self.u = self.rng.uniform(self.umin,self.umax)

    def reset(self):
        self.u = self.rng.uniform(self.umin,self.umax)

    def step(self, obs):
        stepsize = self.rate*self.system.dt
        du = self.rng.uniform(-stepsize,stepsize,self.system.ctrl_dim)
        self.u = np.clip(self.u+du, self.umin, self.umax)
        return self.u

class SinusoidalPolicy(Policy):
    """A policy that evaluates a sinusoid."""
    def __init__(self, system : System, ctrl_bounds : np.ndarray, period = 1.0, frequency = None, phase = 0.0):
        super().__init__(system)
        self.umin, self.umax = ctrl_bounds.T
        self.period = period
        if frequency is not None:
            self.period = 1.0/frequency
        if not hasattr(phase,'__iter__'):
            phase = np.array([phase]*system.ctrl_dim)
        self.phase = phase
        self.mean = (self.umin+self.umax)*0.5
        self.magnitude = self.umax - self.mean
        self.t = 0.0

    def reset(self):
        self.t = 0.0
    
    def step(self,obs):
        return self.mean + self.magnitude * np.sin(2 * np.pi * (self.t / self.period + self.phase))


def uniform_random_generate(system : System, ocp : OCP, dynamics : Dynamics,
        rng, init_min, init_max, 
        traj_len, n_trajs) -> List[Trajectory]:
    """Generate data with a white noise control."""
    if len(init_min) != system.obs_dim or len(init_max) != system.obs_dim:
        raise ValueError("Invalid initial state bounds")
    trajs = []
    policy = WhiteNoisePolicy(system,ocp.get_ctrl_bounds(),rng = rng)
    for _ in range(n_trajs):
        state0 = rng.uniform(init_min, init_max, system.obs_dim)
        trajs.append(simulate(policy,state0,dynamics=dynamics,max_steps=traj_len,silent=True))
    return trajs

def prbs_generate(system : System, ocp : OCP, dynamics : Dynamics,
        rng, init_min, init_max,
        traj_len, n_trajs, ustates, Nswitch):
    """Generate data with a set of piecewise-constant controls,
    sampled from ustates, switching exactly Nswitch times.
    """
    trajs = []
    for _ in range(n_trajs):
        # Compute control sequence
        switches = rng.choice(traj_len, Nswitch)  
        switches = np.concatenate([[0], switches, [traj_len]])
        u = np.zeros((traj_len, system.ctrl_dim))
        for ps, ns in zip(switches[:-1], switches[1:]):
            u[ps:ns] = rng.choice(ustates)
        policy = OpenLoopPolicy(system,u)
        state0 = rng.uniform(init_min, init_max, system.obs_dim)
        trajs.append(simulate(policy,state0,dynamics=dynamics,max_steps=traj_len,silent=True))
    return trajs

def random_walk_generate(system, ocp, dynamics, rng, init_min, init_max, walk_rate,
        traj_len, n_trajs):
    trajs = []
    policy = BrownianNoisePolicy(system,ocp.get_ctrl_bounds(),walk_rate,rng = rng)
    for _ in range(n_trajs):
        state0 = rng.uniform(init_min, init_max, system.obs_dim)
        policy.reset()
        trajs.append(simulate(policy,state0,dynamics=dynamics,max_steps=traj_len,silent=True))
    return trajs

def periodic_control_generate(system, ocp, dynamics, rng, init_min, init_max,
        traj_len, n_trajs):
    trajs = []
    periods = list(range(1, traj_len, max([1, traj_len // n_trajs])))
    print("periods=", periods)
    for period in periods:
        state0 = rng.uniform(init_min, init_max, system.obs_dim)
        policy = SinusoidalPolicy(system,ocp.get_ctrl_bounds(),period*system.dt,phase=rng.choice([0,0.25,0.5,0.75]))
        trajs.append(simulate(policy,state0,dynamics=dynamics,max_steps=traj_len,silent=True))
    return trajs

def multisine_generate(system, task, dynamics, rng, init_min, init_max, n_freqs,
        traj_len, n_trajs, abort_if=None):
    trajs = []
    periods  = list(range(1, traj_len, n_freqs))
    umin, umax = task.get_ctrl_bounds().T
    uamp = (umax - umin) / 2
    umed = (umax + umin) / 2
        
    for _ in range(n_trajs):
        weights = []
        for i in range(system.ctrl_dim):
            vals = rng.uniform(size=len(periods)-1)
            vals = np.concatenate([[0.0], np.sort(vals), [1.0]])
            weight = vals[1:] - vals[:-1]
            weights.append(weight)
        weights = np.array(weights)
        phases = rng.uniform(0, 2*np.pi, len(periods))

        state0 = rng.uniform(init_min, init_max, system.obs_dim)
        y = state0[:]
        traj = Trajectory.zeros(system, traj_len)
        traj.obs[:] = y
        umin, umax = task.get_ctrl_bounds().T
        for i in range(traj_len):
            traj[i].obs[:] = y
            u = np.zeros(system.ctrl_dim)
            for j, period in enumerate(periods):
                u += weights[:,j] * np.cos(2 * np.pi * i / period + phases[j])
            u = uamp * u + umed
            y = dynamics(y, u)
            traj[i].ctrl[:] = u
            if not abort_if is None and abort_if(y):
                traj = traj[:i]
                break
        trajs.append(traj)
    return trajs
