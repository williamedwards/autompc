class TrackingCost(Cost):
    """
    Base class for cost functions.  The general form of a cost is::

        J(x,u) = \int_{t=0}^{T} l(x(t),u(t)) dt + phi(x(T))

    with l(x(t),u(t)) an incremental (running) cost and phi(x(T)) the
    terminal cost.  Note this is a continuous-time formulation; to evaluate
    over a discrete trajectory we use standard Euler integration.


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
        self.system = system
        self._cost = cost
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

    def incremental(self, obs, control, t) -> float:
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
        self._cost.goal = self.goal[t]
        return self._cost.incremental(obs, control)

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
        return self._cost.properties.get('quad',False)

    @property
    def is_convex(self) -> bool:
        """
        True if cost is convex.
        """
        return self._cost.properties.get('convex',False)

    @property
    def is_diff(self) -> bool:
        """
        True if cost is differentiable.
        """
        return self._cost.properties.get('diff',False)

    @property
    def is_twice_diff(self) -> bool:
        """
        True if cost is twice differentiable
        """
        return self._cost.properties.get('twice_diff')

    @property
    def has_goal(self) -> bool:
        """
        True if cost has goal
        """
        return 'goal' in self._cost.properties and self._cost.properties['goal'] is not None
    
    @property
    def goal(self) -> Optional[np.ndarray]:
        """
        Returns the cost goal state if available, or None if no goal is
        available.

        Returns : numpy array of size (traj_len x self.system.obs_dim)
            Goal state
        """
        if 'goal' not in self.properties:
            return None
        return np.copy(self.properties['goal'])

    def set_goal(self,goal):
        """Sets the cost's goal state. (Note: not all costs actually act to
        drive the system toward a goal).
        """
        self.properties['goal'] = np.copy(goal)

    @goal.setter
    def goal(self,goal):
        """Alias for backwards compatibility."""
        self.set_goal(goal)

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