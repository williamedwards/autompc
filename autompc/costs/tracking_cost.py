import numpy as np

from .costs import Cost

class TrackingCost(Cost):
    """
    Time Varying Cost Wrapper
    """
    def __init__(self, system, cost, **properties):
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
        super().__init__(system)
        self._cost = cost
        self.properties = {}

    def update_goal(self, obs):
        self._cost.goal[:-1] = self._cost.goal[1:] 
        self._cost.goal[-1] = obs

    def __call__(self, traj):
        raise NotImplementedError

    def incremental(self, obs, control, t):
        self._cost.set_goal(self.goal[t])
        return self._cost.incremental(obs, control)

    def incremental_diff(self, obs, control, t):
        self._cost.set_goal(self.goal[t])
        return self._cost.incremental_diff(obs, control) 

    def incremental_hess(self, obs, control, t):
        self._cost.set_goal(self.goal[t])
        return self._cost.incremental_diff(obs, control)

    def terminal(self, obs):
        raise NotImplementedError

    def terminal_diff(self, obs):
        raise NotImplementedError

    def terminal_hess(self, obs):
        raise NotImplementedError

    @property
    def is_quad(self):
        """
        True if cost is quadratic.
        """
        return self._cost.properties.get('quad',False)

    @property
    def is_convex(self):
        """
        True if cost is convex.
        """
        return self._cost.properties.get('convex',False)

    @property
    def is_diff(self):
        """
        True if cost is differentiable.
        """
        return self._cost.properties.get('diff',False)

    @property
    def is_twice_diff(self):
        """
        True if cost is twice differentiable
        """
        return self._cost.properties.get('twice_diff')

    @property
    def has_goal(self):
        """
        True if cost has goal
        """
        return 'goal' in self._cost.properties and self._cost.properties['goal'] is not None
    
    def __add__(self, rhs):
        if isinstance(rhs,TrackingCost):
            if (self.goal is None and rhs.goal is None) or np.all(self.goal == rhs.goal):
                return TrackingCost(self.system,self._cost+rhs,self.goal)
        return Cost.__add__(self,rhs)

    def __mul__(self, rhs):
        if not isinstance(rhs,(float,int)):
            raise ValueError("* only supports product with numbers")
        return TrackingCost(self.system,self._cost*rhs,self.goal)
