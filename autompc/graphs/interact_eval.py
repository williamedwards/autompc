# Created by William Edwards

from collections import defaultdict
from pdb import set_trace

import numpy as np
import numpy.linalg as la

from matplotlib.ticker import FormatStrFormatter

from ..grapher import Grapher
from ..graph import Graph


class InteractiveEvalGrapher(Grapher):
    def __init__(self, system):
        super().__init__(system)

    def __call__(self, model, configuration):
        return InteractiveEvalGraph(self.system, model, configuration)

class InteractiveEvalGraph(Graph):
    def __init__(self, system, model, configuration):
        super().__init__(system, model, configuration,
                need_training_eval=True)
        self.testing_trajs = []
        self.training_trajs = []
        self.obs_mins = np.full((system.obs_dim,), np.inf)
        self.obs_maxs = np.full((system.obs_dim,), -np.inf)
        testing_trajs = 0.0
        training_trajs = 0.0
        self.obs_lower_bounds = dict() 
        self.obs_upper_bounds = dict()

    def set_obs_lower_bound(self, obsname, bound):
        self.obs_lower_bounds[obsname] = bound

    def set_obs_upper_bound(self, obsname, bound):
        self.obs_upper_bounds[obsname] = bound

    def add_traj(self, predictor, traj, training=False):
        for i in range(self.system.obs_dim):
            obs_min = np.amin(traj.obs[:,i])
            obs_max = np.amax(traj.obs[:,i])
            if obs_min < self.obs_mins[i]:
                self.obs_mins[i] = obs_min
            if obs_max > self.obs_maxs[i]:
                self.obs_maxs[i] = obs_max
        if not training:
            self.testing_trajs.append((predictor, traj))
        else:
            self.training_trajs.append((predictor, traj))

    def _tabulate(self, preds_and_trajs, obs_lower_bounds, obs_upper_bounds):
        overall_sumsqe = 0.0
        obs_sumsqe = np.zeros((self.system.obs_dim),)
        npoints = 0
        for predictor, traj in preds_and_trajs:
            for i in range(len(traj)-1):
                if (traj[i].obs[:] < obs_lower_bounds).any():
                    continue
                if (traj[i].obs[:] > obs_upper_bounds).any():
                    continue
                predobs = predictor(i, 1)
                overall_sumsqe += la.norm(predobs - traj[i+1].obs[:])**2
                obs_sumsqe += (predobs - traj[i+1].obs[:])**2
                npoints += 1
        return (np.sqrt(np.concatenate([[overall_sumsqe], 
            obs_sumsqe]) / npoints), npoints)

    def __call__(self, fig):
        obs_lower_bounds = np.zeros((self.system.obs_dim,))
        obs_upper_bounds = np.zeros((self.system.obs_dim,))
        for i, obsname in enumerate(self.system.observations):
            if obsname in self.obs_lower_bounds:
                obs_lower_bounds[i] = self.obs_lower_bounds[obsname]
            else:
                obs_lower_bounds[i] = self.obs_mins[i]
            if obsname in self.obs_upper_bounds:
                obs_upper_bounds[i] = self.obs_upper_bounds[obsname]
            else:
                obs_upper_bounds[i] = self.obs_maxs[i]

        testing_res, testing_npoints = self._tabulate(self.testing_trajs,
            obs_lower_bounds, obs_upper_bounds)
        training_res, training_npoints = self._tabulate(self.training_trajs,
            obs_lower_bounds, obs_upper_bounds)

        testing_all_res, testing_all_npoints = self._tabulate(self.testing_trajs,
            self.obs_mins, self.obs_maxs)
        training_all_res, training_all_npoints = self._tabulate(self.training_trajs,
            self.obs_mins, self.obs_maxs)

        n = self.system.obs_dim + 1
        labels = ["Overall"] + self.system.observations
        for i in range(n):
            ax = fig.add_subplot(4, n, i+1)
            ax.set_xlabel("Test " + labels[i])
            ax.set_ylabel("RMSE")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
            ax.set_xticks([])
            ax.bar([1], testing_res[i:i+1])
            ax.set_xlim([0.0, 2.0])
            ax.plot([0.0, 2.0], [testing_all_res[i], testing_all_res[i]], "r-")

        for i in range(n):
            ax = fig.add_subplot(4, n, n+i+1)
            ax.set_xlabel("Train " + labels[i])
            ax.set_ylabel("RMSE")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
            ax.set_xticks([])
            ax.bar([1], training_res[i:i+1], color = "g")
            ax.set_xlim([0.0, 2.0])
            ax.plot([0.0, 2.0], [training_all_res[i], training_all_res[i]], "r-")

        ax = fig.add_subplot(4, n, 2*n+1)
        ax.set_xlabel("Testing points in range")
        sizes = [testing_npoints, testing_all_npoints-testing_npoints]
        ax.pie(sizes, colors=["b", "lightgrey"], labels=list(map(str, sizes)))
        ax = fig.add_subplot(4, n, 2*n+n)
        ax.set_xlabel("Training points in range")
        sizes = [training_npoints, training_all_npoints-training_npoints]
        ax.pie(sizes, colors=["g", "lightgrey"], labels=list(map(str, sizes)))

        for i in range(n-1):
            ax = fig.add_subplot(4, n, 3*n+i+2)
            ax.set_aspect("equal")
            height = 0.25 * (self.obs_maxs[i] - self.obs_mins[i])
            ax.set_xlim([self.obs_mins[i], self.obs_maxs[i]])
            ax.set_ylim([0, height])
            ax.set_yticks([])
            ax.fill_between([obs_lower_bounds[i], obs_upper_bounds[i]],
                    0, height, color="cy")
            ax.set_xlabel(labels[i+1] + " range")
