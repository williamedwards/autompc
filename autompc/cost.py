# Created by William Edwards (wre2@illinois.edu)

class Cost:
    def __call__(self, xs, us, ret_grad=False):
        """
        Returns the cost associated with a particular trajectory
        of xs and us. Returns gradient if ret_grad=True.
        """
        raise NotImplementedError

    def get_terminal(self, x, ret_grad=False):
        """
        Returns terminal cost associated with state x. May
        not be implemented for all cost functions.
        """
        raise NotImplementedError

    def get_additive(self, x, u, ret_grad=False):
        """
        Returns the additive cost for a state x and control u.
        """
        raise NotImplementedError

    def get_quadratic(self):
        """
        Returns cost matrices Q, R, N, F for a cost function of
        the form

        .. math::
            x^T(t_f) F x(t_f) + \sum_i [ x^T(t_i) Q x(t_i) + u^T(t_i) R u(t_i)
                + 2 x^T(t_i) N u(t_i) ]

        Only implemented for quadratic cost functions.
        """
        raise NotImplementedError

