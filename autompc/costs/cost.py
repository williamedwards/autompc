# Created by William Edwards, (wre2@illinois.edu)

import numpy as np

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List,Tuple,Optional

class Cost(ABC):
    """
    Base class for cost functions.  The general form of a cost is::

        J(x,u) = \int_{t=0}^{T} l(x(t),u(t)) dt + phi(x(T))

    with l(x(t),u(t)) an incremental (running) cost and phi(x(T)) the
    terminal cost.  Note this is a continuous-time formulation; to evaluate
    over a discrete trajectory we use standard Euler integration.


    """
    def __init__(self, system, **properties):
        """
        Create cost

        Parameters
        ----------
        system : System
            Robot system for which cost will be evaluated
        properties : Dict
            a dictionary of properties that may be present in a cost and
            relevant to the selection of optimizers. Common values include:
            - 'goal': a goal state (numpy array)
            - 'quad': whether the cost is quadratic (bool)
            - 'convex': whether the cost is convex (bool)
            - 'diff': whether the cost is differentiable (bool)
            - 'twice_diff': whether the cost is twice differentiable (bool)
        """
        self.system = system
        self.properties = {}

    def __call__(self, traj):
        """
        Evaluate cost on whole trajectory, approximating the integral using
        forward Euler.

        Parameters
        ----------
        traj : Trajectory
            Trajectory to evaluate
        """
        cost = 0.0
        for i in range(len(traj)):
            cost += self.incremental(traj[i].obs,traj[i].ctrl)*self.system.dt
        cost += self.terminal(traj[-1].obs)
        return cost

    def incremental(self, obs, control) -> float:
        """
        Evaluates incremental cost at a particular time step.
        Raises exception if not implemented.

        Parameters
        ----------
        obs : self.system.obs_dim
            Observation
        
        ctrl : self.system.ctrl_dim
            Control

        Returns : float
            Cost increment
        """
        raise NotImplementedError

    def incremental_diff(self, obs, ctrl) -> Tuple[float,np.ndarray,np.ndarray]:
        """
        Evaluates the incremental cost at a particular time
        step and computes Jacobians. Raises exception if not
        implemented.

        Returns : (float, numpy array of size self.system.obs_dim, numpy array of size self.system.ctrl_dim)
            Cost, Jacobian dl/dx, Jacobian dl/du
        """
        raise NotImplementedError

    def incremental_hess(self, obs, ctrl) -> Tuple[float,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Evaluates the incremental cost at a particular time
        step and computes Jacobians and Hessians. Raises exception if not
        implemented.

        Returns : (float, numpy array of size self.system.obs_dim,
                  numpy array of size self.system.ctrl_dim,
                  numpy array of shape (self.system.obs_dim, self.system.obs_dim),
                  numpy array of shape (self.system.obs_dim, self.system.ctrl_dim),
                  numpy array of shape (self.system.ctrl_dim, self.system.ctrl_dim)
                  )
            Cost, Jacobian dl/dx, Jacobian dl/du, Hessian d^2l/dx^2,
            Hessian d^2l/dxdu, , Hessian d^2l/du^2
        """
        raise NotImplementedError

    def terminal(self, obs) -> float:
        """
        Evaluates terminal observation cost.
        Raises exception if not implemented.

        Parameters
        ----------
        obs : self.system.obs_dim
            Observation

        Returns : float
            Cost
        """
        raise NotImplementedError

    def terminal_diff(self, obs) -> Tuple[float,np.ndarray]:
        """
        Evaluates the terminal observation cost
        and computes Jacobian. Raises exception if not
        implemented.

        Returns : (float, numpy array of size self.system.obs_dim)
            Cost, Jacobian
        """
        raise NotImplementedError

    def terminal_hess(self, obs) -> Tuple[float,np.ndarray,np.ndarray]:
        """
        Evaluates the terminal observation cost
        and computes Jacobian and Hessian. Raises exception if not
        implemented.

        Returns : (float, numpy array of size self.system.obs_dim, numpy array of shape (self.system.obs_dim, self.system.obsd_im))
            Cost, Jacobian, Hessian
        """
        raise NotImplementedError

    @property
    def is_quad(self) -> bool:
        """
        True if cost is quadratic.
        """
        return self.properties.get('quad',False)

    @property
    def is_convex(self) -> bool:
        """
        True if cost is convex.
        """
        return self.properties.get('convex',False)

    @property
    def is_diff(self) -> bool:
        """
        True if cost is differentiable.
        """
        return self.properties.get('diff',False)

    @property
    def is_twice_diff(self) -> bool:
        """
        True if cost is twice differentiable
        """
        return self.properties.get('twice_diff')

    @property
    def has_goal(self) -> bool:
        """
        True if cost has goal
        """
        return 'goal' in self.properties and self.properties['goal'] is not None
    
    @property
    def goal(self) -> Optional[np.ndarray]:
        """
        Returns the cost goal state if available, or None if no goal is
        available.

        Returns : numpy array of size self.system.obs_dim
            Goal state
        """
        if 'goal' not in self.properties:
            return None
        return np.copy(self.properties['goal'])
    
    @goal.setter
    def goal(self, goal):
        """Sets the cost's goal state. (Note: not all costs actually act to
        drive the system toward a goal).
        """
        self.properties['goal'] = np.copy(goal)

    def set_goal(self,goal):
        """Sets the cost's goal state. (Note: not all costs actually act to
        drive the system toward a goal).
        """
        self.properties['goal'] = np.copy(goal)

    def __add__(self, other):
        if isinstance(other, SumCost):
            return other.__radd__(self)
        else:
            return SumCost(self.system, [self, other])

    def __mul__(self, rhs):
        if not isinstance(rhs,(int,float)):
            raise ValueError("Can only multiply by a float")
        return MulCost(self.system, self, rhs)
        
    def __rmul__(self, lhs):
        if not isinstance(lhs,(int,float)):
            raise ValueError("Can only multiply by a float")
        return MulCost(self.system, self, lhs)


class SumCost(Cost):
    def __init__(self, system, costs : List[Cost]):
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

        if all(cost.is_convex for cost in self.costs):
            self.properties['convex'] = True
        if all(cost.is_diff for cost in self.costs):
            self.properties['diff'] = True
        if all(cost.is_twice_diff for cost in self.costs):
            self.properties['twice_diff'] = True
        goal = None
        if self.costs[0].has_goal:
            goal = self.costs[0].goal
            for cost in self.costs[1:]:
                if not cost.has_goal:
                    goal = None
                    break
                if not (goal == cost.goal).all():
                    goal = None
                    break
            if goal is not None:
                self.properties['goal'] = goal

    @property
    def costs(self) -> List[Cost]:
        return self._costs[:]

    def _sum_results(self, args, attr):
        results = [getattr(cost, attr)(*args) for cost in self.costs]
        if isinstance(results[0], Iterable):
            return [sum(vals) for vals in zip(*results)]
        else:
            return sum(results)

    def incremental(self, obs, ctrl):
        return self._sum_results((obs,ctrl), "incremental")

    def incremental_diff(self, obs, ctrl):
        return self._sum_results((obs,ctrl), "incremental_diff")

    def incremental_hess(self, obs, ctrl):
        return self._sum_results((obs, ctrl), "incremental_hess")

    def terminal(self, obs):
        return self._sum_results((obs,), "terminal")

    def terminal_diff(self, obs):
        return self._sum_results((obs,), "terminal_diff")

    def terminal_hess(self, obs):
        return self._sum_results((obs,), "terminal_hess")

    @property
    def goal(self):
        return super().goal

    @goal.setter
    def goal(self, goal):
        super().goal=goal
        for cost in self.costs:
            cost.goal = goal

    def set_goal(self,goal):
        super().set_goal(goal)
        for cost in self.costs:
            cost.goal = goal

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


class MulCost(Cost):
    def __init__(self, system, cost : Cost, scale : float):
        """
        A cost which is the product of a cost and a number. It can be created using
        the `*` operator

        Parameters
        ----------
        system : System
            System for the cost object.

        cost : Cost
            Base cost object
        
        scale : float
            The amount by which to scale
        """
        super().__init__(system)
        self._cost = cost
        self._scale = scale
        self.properties = cost.properties.copy()

    @property
    def scale(self) -> float:
        return self._scale

    def _mul_results(self, args, attr):
        results = getattr(self._cost, attr)(*args)
        if isinstance(results, Iterable):
            return [(None if val is None else val*self._scale) for val in results]
        else:
            return results*self._scale

    def incremental(self, obs, ctrl):
        return self._mul_results((obs,ctrl), "incremental")

    def incremental_diff(self, obs, ctrl):
        return self._mul_results((obs,ctrl), "incremental_diff")

    def incremental_hess(self, obs, ctrl):
        return self._mul_results((obs, ctrl), "incremental_hess")

    def terminal(self, obs):
        return self._mul_results((obs,), "terminal")

    def terminal_diff(self, obs):
        return self._mul_results((obs,), "terminal_diff")

    def terminal_hess(self, obs):
        return self._mul_results((obs,), "terminal_hess")


    @property
    def goal(self):
        return super().goal

    @goal.setter
    def goal(self, goal):
        super().goal=goal
        for cost in self.costs:
            cost.goal = goal

    def set_goal(self,goal):
        super().set_goal(goal)
        for cost in self.costs:
            cost.goal = goal

    def __mul__(self, scale):
        if not isinstance(scale,(float,int)):
            raise NotImplementedError
        return MulCost(self.system, self._cost, self._scale*scale)
    
    def __rmul__(self, scale):
        if not isinstance(scale,(float,int)):
            raise NotImplementedError
        return MulCost(self.system, self._cost, self._scale*scale)
