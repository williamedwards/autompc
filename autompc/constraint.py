# Created by William Edwards (wre2@illinois.edu)

class Constraint:
    def check(self, xs, us):
        """
        Parameters
        ----------
            xs : Numpy array
                Trajectory states
            us : Numpy Array
                Trajectory controls
        Returns
        -------
            bool
                True if trajectory satisfies constraints,
                false otherwise.
        """
        raise NotImplementedError

    def eval(self, xs, us, ret_grad=False):
        """
        Parameters
        ----------
            xs : Numpy array
                Trajectory states
            us : Numpy array
                Trajectory controls
            ret_grad : bool
                If true, return gradients of numeric constraints,
                if available (otherwise raise NotImplementedErorr).
        Returns 
        -------
            cons_vals : Numpy array
                Must be nonnegative to satisfy.
            cons_grad : Optional, Numpy array
                Gradient of cons_vals wrt xs and us.
        May not be implemented for all constraints.
        """
        raise NotImplementedError

    def get_control_bounds(self):
        """
        Returns list of bounds (lower, upper) for all controls.
        May not be implemented for all constraints.
        """
        raise NotImplementedError

    def check_state(self, x):
        """
        Parameters
        ----------
            x : Numpy array
                State to check.
        Returns 
        -------
            bool 
                True if state satisfies constraint,
                false otherwise.
        May not be implemented for all constraints.
        """
        raise NotImplementedError

    def eval_state(self, x, ret_grad=False):
        """
        Parameters
        ----------
            x : Numpy array
                State to evaluate.
            ret_grad : bool
                If true, return gradients of constraints,
                if available (otherwise raise NotImplementedError).
        Returns 
        -------
            cons_vals : Numpy array
                Must be nonnegative to satisfy.
            cons_grad : Optional, Numpy array
                Gradient of cons_vals wrt x.
        May not be implemented for all constraints.
        """
        raise NotImplementedError

    def get_state_bounds(self):
        """
        Returns list of bounds (lower, upper) for all states.
        May not be implemented for all constraints.
        """
        raise NotImplementedError
