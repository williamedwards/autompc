"""
Implementation of cross entropy method to optimize control sequence with learned model.
It implements both the basic version and improvements from
https://arxiv.org/pdf/2008.06389.pdf

To use parallel computer and vectorized operation, it requires the model function to return a batch prediction.
For simulation based method, it is rather typical to require that to make everything faster.
"""
import numpy as np
import multiprocessing as mp
import copy
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from .controller import Controller
from ..tasks.task import Task


from numpy import sqrt, newaxis
from numpy.fft import irfft, rfftfreq
from numpy.random import normal
from numpy import sum as npsum


"""The following piece of code is directly copy-paste from https://github.com/felixpatzelt/colorednoise/blob/master/colorednoise.py
"""
def powerlaw_psd_gaussian(exponent, size, fmin=0):
    """Gaussian (1/f)**beta noise.
    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)
    Normalised to unit variance
    Parameters:
    -----------
    exponent : float
        The power-spectrum of the generated noise is proportional to
        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2
        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.
    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.
    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.
    Returns
    -------
    out : array
        The samples.
    Examples:
    ---------
    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """
    
    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]
    
    # The number of samples in each time series
    samples = size[-1]
    
    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)
    
    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1./samples) # Low frequency cutoff
    ix   = npsum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)
    
    # Calculate theoretical output standard deviation from scaling
    w      = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2. # correct f = +-0.5
    sigma = 2 * sqrt(npsum(w**2)) / samples
    
    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]
    
    # Generate scaled random power + phase
    sr = normal(scale=s_scale, size=size)
    si = normal(scale=s_scale, size=size)
    
    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2): si[...,-1] = 0
    
    # Regardless of signal length, the DC component must be real
    si[...,0] = 0
    
    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si
    
    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma
    
    return y


class CEM:
    def __init__(self, dyn_eqn, cost_eqn, terminal_cost, model, **kwargs):
        """
        N: number of samples; h: planning horizon; K: size of elite-set; β: colored-noise exponent
        CEM-iterations: number of iterations; γ: reduction factor of samples, σinit: noise strength

        Args:
            dyn_eqn ([type]): The dynamics function which supports batch mode
            cost_eqn ([type]): The integral cost that supports batch evaluation and summation
            terminal_cost ([type]): The terminal cost, batch enabled.
            model ([type]): the basic model, for single step forward. May be different from dyn_eqn
        """
        self.model = model
        self.dyn_eqn = dyn_eqn
        self.cost_eqn = cost_eqn
        self.terminal_cost = terminal_cost
        system = model.system
        self.dim_state, self.dim_ctrl = system.obs_dim, system.ctrl_dim
        self.seed = kwargs.get('seed', 0)
        self.H = kwargs.get('H', 20)
        self.num_path = kwargs.get('num_path', 1000)  # how many paths are generated in parallel
        self.num_elite = kwargs.get('num_elite', 20)  # how many paths are generated in parallel
        self.beta = kwargs.get('beta', 1)  # which color is the noise?
        self.alpha = kwargs.get('alpha', 0.1)  # momentum
        self.num_iter = kwargs.get('niter', 10)
        self.elite_fraction = kwargs.get('elite_fraction', 0.3)  # keep this portion of elites
        self.sigma = kwargs.get('sigma', 1)  # initial noise strength
        self.gamma = kwargs.get('gamma', 1)  # reduction factor for number of samples
        self.mean_act = np.zeros((self.H, self.dim_ctrl))  # the mean action sequence
        ubounds = kwargs.get('ubounds', None)
        if ubounds is None:
            self.umin = self.umax = None
        else:
            self.umin, self.umax = ubounds
        self.elite_set = None  # recorded good stuff from last time

    def do_rollouts(self, cur_state, actions):
        # actions is of shape (sample, udim, horizon)
        n_sample = actions.shape[0]
        path = np.zeros((n_sample, self.dim_state))
        path[:] = cur_state
        costs = np.zeros(n_sample)
        for i in range(self.H):
            if self.umin is None:
                action = actions[:, :, i]
            else:
                action = np.clip(actions[:, :, i], self.umin, self.umax)
                actions[:, :, i] = action
            costs += self.cost_eqn(path, action)
            path = self.dyn_eqn(path, action)
        # the final cost
        if self.terminal_cost:
            costs += self.terminal_cost(path)
        return costs

    def run(self, traj, latent=None):
        # first is to extract current state
        x0 = self.model.traj_to_state(traj)
        # then collect trajectories...
        sigma_i = self.sigma * np.ones((self.dim_ctrl, self.H))
        best_cost = np.inf
        best_elite = None
        for itr_ in range(self.num_iter):
            if self.gamma == 1:
                n_sample = self.num_path
            else:
                n_sample = max(2 * self.num_elite, int(self.num_path * self.gamma ** (-itr_)))
            # collect samples... of noise...
            noises = powerlaw_psd_gaussian(self.beta, (n_sample, self.dim_ctrl, self.H)) * sigma_i[None]
            noises += self.mean_act.T[None]  # make dimension match... and becomes actions
            rd_chs = np.random.choice(self.num_elite, int(self.elite_fraction * self.num_elite))
            # add samples of elites...
            if itr_ == 0:
                if self.elite_set is not None:
                    shifted = np.roll(self.elite_set[rd_chs], -1, axis=1)  # since its elite by h by dimu 
                    noises = np.concatenate((noises, shifted), axis=0)
            else:
                noises = np.concatenate((noises, self.elite_set[rd_chs]), axis=0)
            costs = self.do_rollouts(x0, noises)
            # find the elite set
            cost_order = np.argsort(costs)
            elite_set = noises[cost_order[:self.num_elite]]
            if costs[cost_order[0]] < best_cost:
                best_cost = costs[cost_order[0]]
                best_elite = noises[cost_order[0]]
            # compute mu and sigma (before momentum)
            if self.alpha != 0 and itr_ > 0:
                mu = np.mean(elite_set, axis=0).T * (1 - self.alpha) + self.alpha * mu
                sigma_i = np.std(elite_set, axis=0) * (1 - self.alpha) + self.alpha * sigma_i
            else:
                mu = np.mean(elite_set, axis=0).T
                sigma_i = np.std(elite_set, axis=0)
            self.mean_act = mu
            self.elite_set = elite_set
        # return the first action of the best_elite
        # import pdb; pdb.set_trace()
        # print('sigma_i = ', sigma_i, 'best_cost = ', best_cost)
        ret_action = best_elite.T[0]
        return ret_action, None

    def traj_to_state(self, traj):
        return self.model.traj_to_state(traj)

    @property
    def state_dim(self):
        return self.model.state_dim

    @staticmethod
    def get_configuration_space(system, task, model):
        cs = CS.ConfigurationSpace()
        horizon = CSH.UniformIntegerHyperparameter(name="horizon",
                lower=10, upper=100, default_value=10)
        cs.add_hyperparameter(horizon)
        kappa = CSH.UniformFloatHyperparameter(name='kappa', lower=0.1, upper=1.0, default_value=1.0)
        cs.add_hyperparameter(kappa)
        num_traj = CSH.UniformIntegerHyperparameter(name='num_traj', lower=100, upper=1000, default_value=200)
        cs.add_hyperparameter(num_traj)
        return cs

    @staticmethod
    def is_compatible(system, task, model):
        # MPPI works with all model/system/task unless there is hard constraints, but can't we?
        return True