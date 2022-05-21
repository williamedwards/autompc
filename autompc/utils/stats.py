import numpy as np
from ..trajectory import Trajectory

class StatsAccumulator:
    """A class that allows for online accumulation of statistics of a vector
    distribution.
    """
    def __init__(self, prior = None, prior_strength = None, cov = True):
        self.n = 0
        self.sumw = 0
        self.x = None
        self.x2 = None
        self.xtx = None
        self.use_cov = cov
        self.xmin = None
        self.xmax = None
        if prior is not None:
            self.add(prior,prior_strength)
    
    def add(self,x,weight=1):
        if self.n == 0:
            self.x = weight*x
            self.x2 = weight*np.multiply(x,x)
            if self.use_cov:
                self.xtx = weight*np.outer(x,x)
            self.xmin = np.copy(x)
            self.xmax = np.copy(x)
            self.n = 1
            self.sumw += weight
        else:
            self.n += 1
            self.sumw += weight
            if weight == 1:
                self.x += x
                self.x2 += np.multiply(x,x)
                if self.use_cov:
                    self.xtx += np.outer(x,x)
            else:
                self.x += weight*x
                self.x2 += weight*np.multiply(x,x)
                if self.use_cov:
                    self.xtx += weight*np.outer(x,x)
            self.xmin = np.minimum(x,self.xmin)
            self.xmax = np.maximum(x,self.xmax)
    
    def num(self):
        return self.n

    def mean(self):
        if self.x is None or self.sumw == 0: return self.x
        return self.x / self.sumw
    
    def var(self):
        if self.x is None: return None
        if self.sumw == 0: return np.zeros(self.x.shape)
        xmean = self.mean()
        return self.x2 / self.sumw - np.multiply(xmean,xmean)
    
    def cov(self):
        if not self.use_cov:
            raise RuntimeError("Need to specify cov=True on initialization")
        if self.x is None: return None
        if self.sumw == 0: return np.zeros((self.x.shape[0],self.x.shape[0]))
        xmean = self.mean()
        return self.xtx / self.sumw - np.outer(xmean,xmean)
    
    def std(self):
        if self.x is None: return None
        return np.sqrt(self.var())
    
    def min(self):
        return self.xmin
    
    def max(self):
        return self.xmax

    def __dict__(self):
        return {'mean':self.mean(),'std':self.std(),'var':self.std(),'min':self.min(),'max':self.max()}




class TrajectoryStatsAccumulator:
    def __init__(self,cov=False):
        self.system = None
        self.xs = []
        self.us = []
        self.lens = StatsAccumulator()
        self.use_cov = cov
    
    def add(self,traj : Trajectory, weight=1):
        self.system = traj.system
        for i in range(len(self.xs),len(traj)):
            self.xs.append(StatsAccumulator(cov=self.use_cov))
            self.us.append(StatsAccumulator(cov=self.use_cov))
        for i in range(len(traj)):
            self.xs[i].add(traj.obs[i],weight)
            self.us[i].add(traj.ctrls[i],weight)
        self.lens.add(len(traj),weight)
    
    def num(self):
        return self.lens.num()

    def mean(self):
        return Trajectory(self.system,np.array([x.mean() for x in self.xs]),np.array([u.mean() for u in self.us]))
    
    def min(self):
        return Trajectory(self.system,np.array([x.min() for x in self.xs]),np.array([u.min() for u in self.us]))

    def max(self):
        return Trajectory(self.system,np.array([x.max() for x in self.xs]),np.array([u.max() for u in self.us]))

    def std(self):
        return Trajectory(self.system,np.array([x.std() for x in self.xs]),np.array([u.std() for u in self.us]))

    def var(self):
        return Trajectory(self.system,np.array([x.var() for x in self.xs]),np.array([u.var() for u in self.us]))
    
    def __len__(self):
        return self.lens

    def dict(self):
        return {'num':self.num(),'mean':self.mean(),'std':self.std(),'var':self.var(),'min':self.min(),'max':self.max(),
                'len_mean':self.lens.mean(),'len_std':self.lens.std(),'len_var':self.lens.var(),'len_min':self.lens.min(),'len_max':self.lens.max()}


def traj_stats(trajs):
    """Returns a dictionary of statistics of the trajectories"""
    accum = TrajectoryStatsAccumulator()
    for traj in trajs:
        accum.add(traj)
    return accum.dict()