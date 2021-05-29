# Created by William Edwards (wre2@illinois.edu)

from collections.abc import Iterable

import numpy as np

from .cost import Cost

class SumCost(Cost):
    def __init__(self, system, costs):
        """
        A cost which is the sum of other cost terms. It can be created by combining
        other Cost objects with the `+` operator

        Parameters
        ----------
        system : System
            System for the cost object.

        costs : List of Costs
            Cost objects to be summed.
        """
        super().__init__(system)
        self._costs = costs

    @property
    def costs(self):
        return self._costs[:]

    def get_cost_matrices(self):
        if self.is_quad:
            Q = np.zeros((system.obs_dim, system.obs_dim))
            F = np.zeros((system.obs_dim, system.obs_dim))
            R = np.zeros((system.ctrl_dim, system.ctrl_dim))

            for cost in costs:
                Q_, R_, F_ = cost.get_cost_matrices()
                Q += Q_
                R += R_
                F += F_
            return Q, R, F
        else:
            raise NotImplementedError

    def get_goal(self):
        if self.has_goal:
            return self.costs[0]

    def _sum_results(self, arg, attr):
        results = [getattr(cost, attr)(arg) for cost in self.costs]
        if isinstance(results[0], Iterable):
            return [sum(vals) for vals in zip(*results)]
        else:
            return sum(results)

    def eval_obs_cost(self, obs):
        return self._sum_results(obs, "eval_obs_cost")

    def eval_obs_cost_diff(self, obs):
        return self._sum_results(obs, "eval_obs_cost_diff")

    def eval_obs_cost_hess(self, obs):
        return self._sum_results(obs, "eval_obs_cost_hess")

    def eval_ctrl_cost(self, ctrl):
        return self._sum_results(ctrl, "eval_ctrl_cost")

    def eval_ctrl_cost_diff(self, ctrl):
        return self._sum_results(ctrl, "eval_ctrl_cost_diff")

    def eval_ctrl_cost_hess(self, ctrl):
        return self._sum_results(ctrl, "eval_ctrl_cost_hess")

    def eval_term_obs_cost(self, obs):
        return self._sum_results(obs, "eval_term_obs_cost")

    def eval_term_obs_cost_diff(self, obs):
        return self._sum_results(obs, "eval_term_obs_cost_diff")

    def eval_term_obs_cost_hess(self, obs):
        return self._sum_results(obs, "eval_term_obs_cost_hess")

    @property
    def is_quad(self):
        if not self.costs[0].is_quad:
            return False
        goal = self.costs[0].get_goal()
        for cost in self.costs[1:]:
            if not cost.is_quad:
                return False
            if not (goal == cost.get_goal()).all():
                return False
        return True

    @property
    def is_convex(self):
        for cost in self.costs:
            if not cost.is_convex:
                return False
        return True

    @property
    def is_diff(self):
        for cost in self.costs:
            if not cost.is_diff:
                return False
        return True

    @property
    def is_twice_diff(self):
        for cost in self.costs:
            if not cost.is_diff:
                return False
        return True

    @property
    def has_goal(self):
        if not self.costs[0].has_goal:
            return False
        goal = self.costs[0].get_goal()
        for cost in self.costs[1:]:
            if not cost.has_goal:
                return False
            if not (goal == cost.get_goal()).all():
                return False
        return True

    def __add__(self, other):
        if isinstance(other, SumCost):
            return SumCost(self.system, [*self.costs, *other.costs])
        else:
            return SumCost(self.system, [*self.costs, other])

    def __radd__(self, other):
        if isinstance(other, SumCost):
            return SumCost(self.system, [*other.costs, *self.costs])
        else:
            return SumCost(self.system, [other, *self.costs])
