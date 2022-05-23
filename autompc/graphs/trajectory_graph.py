import copy
from typing import List
import matplotlib.pyplot as plt

from ..trajectory import Trajectory
from ..policy import Policy
from ..dynamics import Dynamics
from ..utils.simulation import rollout,simulate

def _normalize_dims(sys,dims):
    if dims is None:
        dims = range(sys.obs_dim)
    elif len(dims) > 0:
        dims = [d if isinstance(d,int) else sys.observations.index(d) for i,d in enumerate(dims)]
        for d in dims:
            if d < 0 or d >= sys.obs_dim:
                raise ValueError("Invalid dimension specified")
    return dims

def _normalize_opts(sys,obs_opts,ctrl_opts):
    if obs_opts is not None:
        if isinstance(obs_opts,bool):
            pass
        elif isinstance(obs_opts,dict):
            obs_opts = [obs_opts.copy() for i in range(sys.obs_dim)]
        else:
            if len(obs_opts) != sys.obs_dim:
                raise ValueError("Need observation options to match # of observations")
            obs_opts = [o.copy() for o in obs_opts]
    else:
        obs_opts = [dict() for i in range(sys.obs_dim)]
    if ctrl_opts is not None:
        if isinstance(ctrl_opts,bool):
            pass
        elif isinstance(ctrl_opts,dict):
            ctrl_opts = [ctrl_opts.copy() for i in range(sys.ctrl_dim)]
        else:
            if len(ctrl_opts) != sys.ctrl_dim:
                raise ValueError("Need control options to match # of controls")
            ctrl_opts = [o.copy() for o in ctrl_opts]
    else:
        ctrl_opts = [dict() for i in range(sys.ctrl_dim)]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if obs_opts != False:
        for dim,o in enumerate(obs_opts):
            if 'label' not in obs_opts[dim]:    
                o['label'] = sys.observations[dim]
            if 'color' not in o:
                o['color'] = colors[dim%len(colors)]
    if ctrl_opts != False:
        for dim,o in enumerate(ctrl_opts):
            if 'label' not in ctrl_opts[dim]:
                o['label'] = sys.controls[dim]
            if 'color' not in o:
                o['color'] = colors[(dim+sys.obs_dim)%len(colors)]
    return obs_opts,ctrl_opts

def plot_traj(traj : Trajectory, obs_opts=None, ctrl_opts=None, dims=None, ax=None):
    """Plots a trajectory's observations and controls over time.

    Arguments
    ----------
        traj : Trajectory
            The trajectory to plot
        obs_opts: None, dict, list of dict, or False
            matplotlib options for drawing the observations of this trajectory
        ctrl_opts: None, list of dict, or False
            matplotlib options for drawing the controls of this trajectory
        dims: optional List[int] or List[str]
            The observation dimensions to plot.
    """
    sys = traj.system
    if ax is None:
        ax = plt.gca()
    dims = _normalize_dims(sys,dims)
    obs_opts, ctrl_opts = _normalize_opts(sys,obs_opts,ctrl_opts)

    if obs_opts:
        for dim in dims:
            ax.plot(traj.times,traj.obs[:,dim],**obs_opts[dim])
    if ctrl_opts:
        for dim in range(sys.ctrl_dim):
            ax.plot(traj.times,traj.ctrls[:,dim],**ctrl_opts[dim])
    ax.set_xlabel('t (s)')
    ax.legend()

def plot_traj_projected(traj : Trajectory, dims=[0,1], marker_rate='auto',opts=None, ax=None):
    """Plots a trajectory projected onto an x-y plane.

    Arguments
    ---------
        traj : Trajectory
            The trajectory to plot
        dims: List[int] or List[str]:
            The observation dimensions to plot.
    """
    if len(dims) != 2:
        raise ValueError("Can only project 2 dimensions, for now")
    sys = traj.system
    if ax is None:
        ax = plt.gca()
    if opts is None:
        opts = dict()
    else:
        opts = opts.copy()
    if marker_rate=='auto':
        import math
        marker_rate = int(math.ceil(len(traj)/20))
    dims = [d if isinstance(d,int) else sys.observations.index(d) for d in dims]
    for d in dims:
        if dims[d] < 0 or dims[d] >= sys.obs_dim:
            raise ValueError("Invalid dimension specified")
    if 'color' not in opts:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        opts['color'] = colors[0]
    ax.plot(traj.obs[:,dims[0]],traj.obs[:,dims[1]],**opts)
    if marker_rate > 0:
        ax.scatter(traj.obs[::marker_rate,dims[0]],traj.obs[::marker_rate,dims[1]],**opts)
    ax.scatter([traj.obs[0,dims[0]]],[traj.obs[0,dims[1]]],marker='s',**opts)
    ax.set_xlabel(sys.observations[dims[0]])
    ax.set_ylabel(sys.observations[dims[1]])

def plot_rollout(traj : Trajectory, dynamics: Dynamics, compare=True, obs_opts=None, ctrl_opts=None, dims=None, ax=None):
    traj_dyn = rollout(dynamics,traj)
    plot_traj(traj_dyn, obs_opts=obs_opts, ctrl_opts=ctrl_opts, dims=dims, ax=ax)
    if compare:
        orig_obs_opts,orig_ctrl_opts = _normalize_opts(traj.system,obs_opts,ctrl_opts)
        for i,o in enumerate(orig_obs_opts):
            o['linestyle'] = '--'
            o['label'] = o['label'] + ' actual'
        plot_traj(traj, obs_opts=orig_obs_opts, ctrl_opts=False, dims=dims, ax=ax)

def plot_simulation(policy : Policy, init_obs, dynamics: Dynamics, max_steps=100, term_cond=None,
    obs_opts=None, ctrl_opts=None, dims=None, ax=None):
    traj = simulate(policy,init_obs,dynamics,term_cond, max_steps=max_steps, silent=True)
    plot_traj(traj, obs_opts=obs_opts, ctrl_opts=ctrl_opts, dims=dims, ax=ax)

def plot_trajs(trajs : List[Trajectory], style='auto', 
                obs_opts=None, ctrl_opts=None, dims=None, ax=None):
    """Plots multiple trajectories.

    Plot style is determined by the `style` argument:
    - 'all': just plots all trajectories on the same plot.
    - 'mean': just plots the mean.
    - 'confidence': plots mean +/- std.
    - 'range': plots mean, min, and max.
    - 'auto' auto-determines from the set size. If size * dims <= 10, uses 'all', otherwise 'confidence'.
    """
    if len(trajs)==0:
        return
    sys = trajs[0].system
    if style == 'auto':
        if dims is None:
            nitems = len(trajs)*sys.obs_dim
        else:
            nitems = len(trajs)*len(dims)
        if nitems <= 10:
            style = 'all'
        else:
            style = 'confidence'
    if style == 'all':
        for traj in trajs:
            plot_traj(traj,obs_opts,ctrl_opts, dims, ax)
    elif style in ['confidence','range','mean']:
        from ..utils.stats import traj_stats
        stats = traj_stats(trajs)
        tmean = stats['mean']
        tstd = stats['std']
        plot_traj(tmean, obs_opts=obs_opts, ctrl_opts=ctrl_opts, dims=dims, ax=ax)
        if style == 'mean':
            return
        elif style == 'confidence':
            xlower = tmean.obs - tstd.obs
            xupper = tmean.obs + tstd.obs
            ulower = tmean.ctrls - tstd.ctrls
            uupper = tmean.ctrls + tstd.ctrls
        else:
            xlower = stats['min'].obs
            xupper = stats['max'].obs
            ulower = stats['min'].ctrls
            uupper = stats['max'].ctrls
        dims = _normalize_dims(sys,dims)
        obs_opts,ctrl_opts = _normalize_opts(sys,obs_opts,ctrl_opts)
        if obs_opts != False:
            for i,o in enumerate(obs_opts):
                if 'alpha' in o:
                    o['alpha'] = 0.1*o['alpha']
                else:
                    o['alpha'] = 0.1
        if ctrl_opts != False:
            for i,o in enumerate(ctrl_opts):
                if 'alpha' in o:
                    o['alpha'] = 0.1*o['alpha']
                else:
                    o['alpha'] = 0.1
        if ax is None:
            ax = plt.gca()
        if obs_opts != False:
            for dim in dims:
                ax.fill_between(tmean.times,xlower[:,dim],xupper[:,dim],**obs_opts[dim])
        if ctrl_opts != False:
            for dim in range(sys.ctrl_dim):
                ax.fill_between(tmean.times,ulower[:,dim],uupper[:,dim],**obs_opts[dim])
    else:
        raise ValueError("Invalid style specified")

def plot_trajs_projected(trajs : List[Trajectory], style='auto', dims=[0,1], marker_rate='auto',
                opts=None, ax=None):
    """Plots multiple trajectories.

    Plot style is determined by the `style` argument:
    - 'all': just plots all trajectories on the same plot.
    - 'mean': plots mean trajectory.
    - 'sample': samples up to 10 tarjectories
    - 'auto' auto-determines from the set size. If size <= 10, uses 'all', otherwise 'sample'.
    """
    if len(trajs)==0:
        return
    sys = trajs[0].system
    if style == 'auto':
        if len(trajs) <= 10:
            style = 'all'
        else:
            style = 'mean'
    if style == 'all':
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        if opts is not None:
            opts = opts.copy()
        else:
            opts = dict()
        for i,traj in enumerate(trajs):
            if 'color' not in opts:
                opts['color'] = colors[i%len(colors)]
            plot_traj_projected(traj, dims, marker_rate=marker_rate, opts=opts, ax=ax)
    elif style == 'sample':
        if len(trajs) < 10:
            return plot_trajs_projected(trajs,'all',dims,marker_rate,opts,ax)
        else:
            import random
            sample = list(random.sample(trajs,10))
            return plot_trajs_projected(sample,'all',dims,marker_rate,opts,ax)
    elif style == 'mean':
        from ..utils.stats import traj_stats
        stats = traj_stats(trajs)
        tmean = stats['mean']
        plot_traj_projected(tmean, dims, marker_rate=marker_rate, opts=opts, ax=ax)
    else:
        raise ValueError("Invalid style specified")