"""
This file implement iterative LQR methods for optimization over nonlinear models.
But after realizing ilqr is different from optimization + local lqr feedback.
I start questioning myself whether this is gonna work, but anyway... It won't hurt.
"""
import numpy as np
import numpy.linalg as la
from pdb import set_trace
from control.matlab import dare

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from ..controller import Controller


class IterativeLQR(Controller):
    def __init__(self, system, task, model, horizon, reuse_feedback=-1, ubounds=None, mode=None, verbose=False):
        """Reuse_feedback determines how many steps of K are used as feedback.
        ubounds is a tuple of minimum and maximum control bounds
        mode specifies mode, 'barrier' use barrier method for control bounds; 'auglag' use augmented Lagrangian; None use default one, clip
        """
        super().__init__(system, task, model)
        self.horizon = horizon
        self.dt = system.dt
        self._need_recompute = True  # this indicates a new iteration is required...
        self._step_count = 0
        if reuse_feedback is None or reuse_feedback <= 0:
            self.reuse_feedback = 0
        elif reuse_feedback > horizon:
            self.reuse_feedback = horizon
        else:
            self.reuse_feedback = reuse_feedback
        self._guess = None
        if ubounds is None and task.are_ctrl_bounded():
            bounds = task.get_ctrl_bounds()
            self.ubounds = (bounds[:,0], bounds[:,1])
        else:
            self.ubounds = ubounds
        self.mode = mode
        self.verbose = verbose
        if mode is None:
            self.compute_ilqr = self.compute_ilqr_default
        elif mode == 'barrier':
            self.compute_ilqr = self.compute_barrier_ilqr
        elif mode == 'auglag':
            self.compute_ilqr = self.compute_auglag_ilqr
        else:
            raise Exception("mode has to be None/barrier/auglag")

    @property
    def state_dim(self):
        return self.model.state_dim + self.system.ctrl_dim

    @staticmethod
    def get_configuration_space(system, task, model):
        cs = ConfigurationSpace()
        horizon = UniformIntegerHyperparameter(name="horizon",
                lower=1, upper=1000, default_value=10)
        cs.add_hyperparameter(horizon)
        return cs

    @staticmethod
    def is_compatible(system, task, model):
        return (task.is_cost_quad()
                and not task.are_obs_bounded()
                and not task.are_ctrl_bounded()
                and not task.eq_cons_present()
                and not task.ineq_cons_present())
 
    def traj_to_state(self, traj):
        return self.model.traj_to_state(traj)

    def compute_auglag_ilqr(self, state, uguess, lmd0=None, mu0=None, phi=None, u_threshold=1e-3, ls_max_iter=5, ls_discount=0.2, ls_cost_threshold=0.1):
        """This one considers control bounds with augmented Lagrangian. It's similar to barrier one,
        just with different linearization and update scheme. It's possible to design a unified interface but why bother.
        https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf
        """
        assert self.ubounds is not None
        umin, umax = self.ubounds
        Q, R, F = self.task.get_cost().get_cost_matrices()
        Q = Q * self.system.dt
        R = R * self.system.dt
        H = self.horizon

        # this function actually evaluates Lagrangian...
        # lmd is of shape (2, horizon, nu), same with mu...
        def eval_obj(xs, us, lmd, imu):
            obj = 0
            for i in range(H):
                obj += xs[i] @ Q @ xs[i] + us[i] @ R @ us[i]
            obj += xs[-1] @ F @ xs[-1]
            obj *= 0.5  # get the half...
            c1 = us - umax
            c2 = umin - us  # both negative...
            obj += np.sum(c1 * lmd[0]) + np.sum(c2 * lmd[1]) + 0.5 * np.sum(imu[0] * c1 ** 2 + imu[1] * c2 ** 2)
            return obj

        # given a single ui, compute gradient and Hessian of
        # 1/2 u^T R u + lmd1 (u - umax) + lmd2 (umin - u) + quadratic term
        def grad_hessian_control(u, R, lmd, imu):  
            c1 = u - umax
            c2 = umin - u  # both negative...
            grad = R @ u + lmd[0] - lmd[1] + imu[0] * c1 - imu[1] * c2
            hess = R + np.diag(imu[0] + imu[1])
            return grad, hess

        dimx, dimu = self.system.obs_dim, self.system.ctrl_dim
        if lmd0 is None:
            lmd0 = np.zeros((2, H, dimu))
        if mu0 is None:
            mu0 = 1
        if phi is None:
            phi = 5
        muf = mu0 * phi ** 10  # a huge number...
        # handy variables...
        states, new_states = np.zeros((2, H + 1, dimx))
        ctrls, new_ctrls = np.zeros((2, H, dimu))
        Ks = np.zeros((H, dimu, dimx))
        ks = np.zeros((H, dimu))
        Jacs = np.zeros((H, dimx, dimx + dimu))  # Jacobian from dynamics...
        cur_lmd = lmd0.copy()
        cur_imu = np.zeros_like(cur_lmd)
        cur_mu = mu0
        c_val = np.zeros((2, H, dimu))
        # first forward simulation
        states[0] = state
        ctrls[:] = uguess
        c_val[0] = ctrls - umax
        c_val[1] = umin - ctrls
        solve_success = False
        for i in range(H):
            states[i + 1], jx, ju = self.model.pred_diff(states[i], ctrls[i])
            Jacs[i, :, :dimx] = jx
            Jacs[i, :, dimx:] = ju
        # first loop on mu
        while cur_mu <= muf:
            # set imu by value of c_val and lmd
            print("cur_mu={}, muf={}".format(cur_mu, muf))
            cur_imu[:] = cur_mu
            cur_imu[(cur_lmd == 0) & (c_val < -1e-6)] = 0  # NOTE: it's important to use &, otherwise it won't converge...
            obj = eval_obj(states, ctrls, cur_lmd, cur_imu)
            initcost = obj
            # start inner iteration from here
            max_iter = 100
            Ct = np.zeros((dimx + dimu, dimx + dimu))
            ct = np.zeros(dimx + dimu)
            converged = False
            du_norm = np.inf
            for itr in range(max_iter):
                # compute at the last step, Vn and vn, just hessian and gradient at the last state
                print("Iteration {}/{}".format(itr, max_iter))
                Vn = F
                vn = F @ states[H]
                lin_cost_reduce = quad_cost_reduce = 0
                for t in range(H, 0, -1):  # so run for H steps...
                    # first assemble Ct and ct, they are linearized at current state
                    Ct[:dimx, :dimx] = Q
                    ct[:dimx] = Q @ states[t - 1]
                    ct[dimx:], Ct[dimx:, dimx:] = grad_hessian_control(ctrls[t - 1], R, cur_lmd[:, t - 1, :], cur_imu[:, t - 1, :])
                    Qt = Ct + Jacs[t - 1].T @ Vn @ Jacs[t - 1]
                    qt = ct + Jacs[t - 1].T @ vn
                    # ready to compute feedback
                    # Qtinv = np.linalg.inv(Qt)
                    Ks[t - 1] = -np.linalg.solve(Qt[dimx:, dimx:], Qt[dimx:, :dimx])
                    ks[t - 1] = -np.linalg.solve(Qt[dimx:, dimx:], qt[dimx:])
                    lin_cost_reduce += qt[dimx:].dot(ks[t - 1])
                    quad_cost_reduce += ks[t - 1] @ Qt[dimx:, dimx:] @ ks[t - 1]
                    # update Vn and vn
                    Vn = Qt[:dimx, :dimx] + Qt[:dimx, dimx:] @ Ks[t - 1] + Ks[t - 1].T @ Qt[dimx:, :dimx] + Ks[t - 1].T @ Qt[dimx:, dimx:] @ Ks[t - 1]
                    vn = qt[:dimx] + Qt[:dimx, dimx:] @ ks[t - 1] + Ks[t - 1].T @ (qt[dimx:] + Qt[dimx:, dimx:] @ ks[t - 1])
                # redo forward simulation and record actions...
                new_states[0] = state
                ls_success = False
                ls_alpha = 1
                best_alpha = None
                best_obj = np.inf
                best_obj_estimate_reduction = None
                ks_norm = np.linalg.norm(ks)
                # reduce alpha until forward simulation only gets control within bounds
                for lsitr in range(ls_max_iter):
                    ctrl_in_bounds = True
                    for i in range(H):
                        new_ctrls[i] = ls_alpha * ks[i] + ctrls[i] + Ks[i] @ (new_states[i] - states[i])
                        new_states[i + 1] = self.model.pred(new_states[i], new_ctrls[i])
                    if not ctrl_in_bounds:
                        ls_alpha *= ls_discount
                        continue
                    new_obj = eval_obj(new_states, new_ctrls, cur_lmd, cur_imu)
                    expect_cost_reduction = ls_alpha * lin_cost_reduce + ls_alpha ** 2 * quad_cost_reduce / 2
                    if (obj - new_obj) / (-expect_cost_reduction) > ls_cost_threshold:
                        best_obj = new_obj
                        best_alpha = ls_alpha
                        break
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_alpha = ls_alpha
                    ls_alpha *= ls_discount
                    if ks_norm < u_threshold:
                        break
                # print('line search obj %f to %f at alpha = %f' % (obj, new_obj, ls_alpha))
                if best_obj < obj or ks_norm < u_threshold:
                    ls_success = True
                    for i in range(H):
                        new_ctrls[i] = best_alpha * ks[i] + ctrls[i] + Ks[i] @ (new_states[i] - states[i])
                        new_states[i + 1], jx, ju = self.model.pred_diff(new_states[i], new_ctrls[i])
                        Jacs[i, :, :dimx] = jx
                        Jacs[i, :, dimx:] = ju
                    new_obj = best_obj
                if (not ls_success and best_obj > obj + 1e-3) or best_alpha is None:
                    print('Line search fails...')
                    break
                else:
                    pass
                    # print('alpha is successful at %f with cost from %f to %f' % (best_alpha, obj, new_obj))
                # return since update of action is small
                # print('u update', np.linalg.norm(new_ctrls - ctrls))
                du_norm = np.linalg.norm(new_ctrls - ctrls)
                if du_norm < u_threshold:
                    # print('Break since update of control is small at %f' % (np.linalg.norm(new_ctrls - ctrls)))
                    converged = True
                # ready to swap...
                states, new_states = new_states, states
                ctrls, new_ctrls = new_ctrls, ctrls
                c_val[0] = ctrls - umax
                c_val[1] = umin - ctrls
                obj = new_obj
                if converged:
                    print('Convergence achieved within %d iterations' % itr)
                    print('Cost update from %f to %f' % (initcost, obj))
                    print('Final state is ', states[-1])
                    break
            if not converged:
                print('ilqr fails to converge, try a new guess? Last u update is %f ks norm is %f' % (du_norm, ks_norm))
                print('ilqr is not converging..., return now...')
                return converged, states, ctrls, Ks, ks
            else:
                # check if all constraints are satisfied
                if np.all(c_val < 1e-3):
                    solve_success = True  # succeed only when ilqr converges and constraints are satisfied
                    print('Nice! All constraints satisfied, exit...')
                    break
                print('ilqr converges for this mu, updating multipliers...')
                cur_lmd = np.maximum(0, cur_lmd + cur_mu * c_val)
                cur_mu *= phi
        return solve_success, states, ctrls, Ks, ks

    def compute_barrier_ilqr(self, state, uguess, mu0=1e-2, muf=1e-2, mu_scale=0.1, u_threshold=1e-3, ls_max_iter=10, ls_discount=0.2, ls_cost_threshold=0.1):
        """This one considers control bounds by imposing log barrier.
        mu0 is the initital penalty
        muf is the final penalty
        mu_scale is how we update penalty
        """
        assert self.ubounds is not None
        umin, umax = self.ubounds
        Q, R, F = self.task.get_cost().get_cost_matrices()
        Q = Q * self.system.dt
        R = R * self.system.dt
        H = self.horizon

        def eval_obj(xs, us, mu):
            obj = 0
            for i in range(H):
                obj += xs[i] @ Q @ xs[i] + us[i] @ R @ us[i]
            obj += xs[-1] @ F @ xs[-1]
            us1 = us - umin
            us2 = umax - us
            barrier = mu * np.sum(np.log(us1) + np.log(us2))
            return obj * 0.5 - barrier

        # given a single ui, compute gradient and Hessian of
        # 1/2 u^T R u - mu log(ui - umin) - mu log(umax - ui)
        def grad_hessian_control(u, R, mu):  
            us1 = u - umin
            us2 = umax - u
            inv1 = 1 / us1
            inv2 = 1 / us2
            grad = R @ u - mu * inv1 + mu * inv2
            hess = R + np.diag(mu * inv1 ** 2 + mu * inv2 ** 2)
            return grad, hess

        dimx, dimu = self.system.obs_dim, self.system.ctrl_dim
        # handy variables...
        states, new_states = np.zeros((2, H + 1, dimx))
        ctrls, new_ctrls = np.zeros((2, H, dimu))
        Ks = np.zeros((H, dimu, dimx))
        ks = np.zeros((H, dimu))
        Jacs = np.zeros((H, dimx, dimx + dimu))  # Jacobian from dynamics...
        cur_mu = mu0
        # first forward simulation
        states[0] = state
        ctrls[:] = uguess
        for i in range(H):
            states[i + 1], jx, ju = self.model.pred_diff(states[i], ctrls[i])
            Jacs[i, :, :dimx] = jx
            Jacs[i, :, dimx:] = ju
        # first loop on mu
        while cur_mu >= muf:
            print("cur_mu={}, muf={}".format(cur_mu, muf))
            obj = eval_obj(states, ctrls, cur_mu)
            initcost = obj
            # start iteration from here
            max_iter = 100
            Ct = np.zeros((dimx + dimu, dimx + dimu))
            ct = np.zeros(dimx + dimu)
            converged = False
            du_norm = np.inf
            for itr in range(max_iter):
                # compute at the last step, Vn and vn, just hessian and gradient at the last state
                print("Iteration {} / {}".format(itr, max_iter))
                Vn = F
                vn = F @ states[H]
                lin_cost_reduce = quad_cost_reduce = 0
                for t in range(H, 0, -1):  # so run for H steps...
                    # first assemble Ct and ct, they are linearized at current state
                    Ct[:dimx, :dimx] = Q
                    ct[:dimx] = Q @ states[t - 1]
                    ct[dimx:], Ct[dimx:, dimx:] = grad_hessian_control(ctrls[t - 1], R, cur_mu)
                    Qt = Ct + Jacs[t - 1].T @ Vn @ Jacs[t - 1]
                    qt = ct + Jacs[t - 1].T @ (vn)  # here Vn @ states[t] may be necessary
                    # ready to compute feedback
                    # Qtinv = np.linalg.inv(Qt)
                    Ks[t - 1] = -np.linalg.solve(Qt[dimx:, dimx:], Qt[dimx:, :dimx])
                    ks[t - 1] = -np.linalg.solve(Qt[dimx:, dimx:], qt[dimx:])
                    lin_cost_reduce += qt[dimx:].dot(ks[t - 1])
                    quad_cost_reduce += ks[t - 1] @ Qt[dimx:, dimx:] @ ks[t - 1]
                    # Ks[t - 1] = -Qtinv[dimx:, dimx:] @ Qt[dimx:, :dimx]
                    # ks[t - 1] = -Qtinv[dimx:, dimx:] @ qt[dimx:]
                    # update Vn and vn
                    Vn = Qt[:dimx, :dimx] + Qt[:dimx, dimx:] @ Ks[t - 1] + Ks[t - 1].T @ Qt[dimx:, :dimx] + Ks[t - 1].T @ Qt[dimx:, dimx:] @ Ks[t - 1]
                    vn = qt[:dimx] + Qt[:dimx, dimx:] @ ks[t - 1] + Ks[t - 1].T @ (qt[dimx:] + Qt[dimx:, dimx:] @ ks[t - 1])
                # redo forward simulation and record actions...
                new_states[0] = state
                ls_success = False
                ls_alpha = 1
                best_alpha = None
                best_obj = np.inf
                best_obj_estimate_reduction = None
                ks_norm = np.linalg.norm(ks)
                # reduce alpha until forward simulation only gets control within bounds
                for lsitr in range(ls_max_iter):
                    ctrl_in_bounds = True
                    for i in range(H):
                        new_ctrls[i] = ls_alpha * ks[i] + ctrls[i] + Ks[i] @ (new_states[i] - states[i])
                        if np.any(new_ctrls[i] >= umax) or np.any(new_ctrls[i] <= umin):
                            ctrl_in_bounds = False
                            break
                        new_states[i + 1] = self.model.pred(new_states[i], new_ctrls[i])
                    if not ctrl_in_bounds:
                        ls_alpha *= ls_discount
                        continue
                    new_obj = eval_obj(new_states, new_ctrls, cur_mu)
                    expect_cost_reduction = ls_alpha * lin_cost_reduce + ls_alpha ** 2 * quad_cost_reduce / 2
                    if (obj - new_obj) / (-expect_cost_reduction) > ls_cost_threshold:
                        best_obj = new_obj
                        best_alpha = ls_alpha
                        break
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_alpha = ls_alpha
                    ls_alpha *= ls_discount
                    if ks_norm < u_threshold:
                        break
                # print('line search obj %f to %f at alpha = %f' % (obj, new_obj, ls_alpha))
                if best_obj < obj or ks_norm < u_threshold:
                    ls_success = True
                    for i in range(H):
                        new_ctrls[i] = best_alpha * ks[i] + ctrls[i] + Ks[i] @ (new_states[i] - states[i])
                        new_states[i + 1], jx, ju = self.model.pred_diff(new_states[i], new_ctrls[i])
                        Jacs[i, :, :dimx] = jx
                        Jacs[i, :, dimx:] = ju
                    new_obj = best_obj
                if (not ls_success and best_obj > obj + 1e-3) or best_alpha is None:
                    print('Line search fails...')
                    break
                else:
                    pass
                    # print('alpha is successful at %f with cost from %f to %f' % (best_alpha, obj, new_obj))
                # return since update of action is small
                # print('u update', np.linalg.norm(new_ctrls - ctrls))
                du_norm = np.linalg.norm(new_ctrls - ctrls)
                if du_norm < u_threshold:
                    # print('Break since update of control is small at %f' % (np.linalg.norm(new_ctrls - ctrls)))
                    converged = True
                # ready to swap...
                states, new_states = new_states, states
                ctrls, new_ctrls = new_ctrls, ctrls
                obj = new_obj
                if converged:
                    print('Convergence achieved within %d iterations' % itr)
                    print('Cost update from %f to %f' % (initcost, obj))
                    print('Final state is ', states[-1])
                    break
            if not converged:
                print('ilqr fails to converge, try a new guess? Last u update is %f ks norm is %f' % (du_norm, ks_norm))
                print('ilqr is not converging..., return now...')
                return converged, states, ctrls, Ks, ks
            else:
                print('ilqr converges for this mu, updating mu...')
                cur_mu *= mu_scale
        return True, states, ctrls, Ks, ks

    def compute_ilqr_default(self, state, uguess, u_threshold=1e-3, ls_max_iter=10, ls_discount=0.2, ls_cost_threshold=0.3):
        """Use equations from https://medium.com/@jonathan_hui/rl-lqr-ilqr-linear-quadratic-regulator-a5de5104c750 .
        A better version is https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
        Here I do not have Hessian correction since I'm certain all my matrices are SPD
        """
        cost = self.task.get_cost()
        # Cost function example
        # cost = cost.eval_obs_cost(obs)
        # cost, cost_jac = cost.eval_obs_cost_diff(obs)
        # cost, cost_jac, cost_hess = cost.eval_obs_cost_hess(obs)
        # cost = cost.eval_ctrl_cost(ctrl)
        # cost, cost_jac = cost.eval_ctrl_cost_diff(ctrl)
        # cost, cost_jac, cost_hess = cost.eval_ctrl_cost_hess(ctrl)
        # cost = cost.eval_term_obs_cost(obs)
        # cost, cost_jac = cost.eval_term_obs_cost_diff(obs)
        # cost, cost_jac, cost_hess = cost.eval_term_obs_cost_hess(obs)
        Q, R, F = cost.get_cost_matrices()
        x0 = cost.get_x0()
        Q = Q * self.system.dt
        R = R * self.system.dt
        H = self.horizon

        def eval_obj(xs, us):
            obj = 0
            for i in range(H):
                obj += xs[i] @ Q @ xs[i] + us[i] @ R @ us[i]
            obj += xs[-1] @ F @ xs[-1]
            return obj * 0.5

        dimx, dimu = self.system.obs_dim, self.system.ctrl_dim
        # handy variables...
        states, new_states = np.zeros((2, H + 1, dimx))
        ctrls, new_ctrls = np.zeros((2, H, dimu))
        ls_states = np.zeros((ls_max_iter, H + 1, dimx))
        ls_ctrls = np.zeros((ls_max_iter, H, dimu))
        Ks = np.zeros((H, dimu, dimx))
        ks = np.zeros((H, dimu))
        Jacs = np.zeros((H, dimx, dimx + dimu))  # Jacobian from dynamics...
        # first forward simulation
        states[0] = state
        ctrls[:] = uguess
        for i in range(H):
            states[i + 1], jx, ju = self.model.pred_diff(states[i], ctrls[i])
            Jacs[i, :, :dimx] = jx
            Jacs[i, :, dimx:] = ju
        obj = eval_obj(states, ctrls)
        initcost = obj
        # start iteration from here
        max_iter = 100
        Ct = np.zeros((dimx + dimu, dimx + dimu))
        ct = np.zeros(dimx + dimu)
        converged = False
        du_norm = np.inf
        for itr in range(max_iter):
            if self.verbose:
                print('At iteration %d' % itr)
            # compute at the last step, Vn and vn, just hessian and gradient at the last state
            Vn = F
            vn = F @ states[H]
            lin_cost_reduce = quad_cost_reduce = 0
            for t in range(H, 0, -1):  # so run for H steps...
                # first assemble Ct and ct, they are linearized at current state
                Ct[:dimx, :dimx] = Q
                Ct[dimx:, dimx:] = R
                ct[:dimx] = Q @ states[t - 1]
                ct[dimx:] = R @ ctrls[t - 1]
                Qt = Ct + Jacs[t - 1].T @ Vn @ Jacs[t - 1]
                qt = ct + Jacs[t - 1].T @ (vn)  # here Vn @ states[t] may be necessary
                # ready to compute feedback
                # Qtinv = np.linalg.inv(Qt)
                Ks[t - 1] = -np.linalg.solve(Qt[dimx:, dimx:], Qt[dimx:, :dimx])
                ks[t - 1] = -np.linalg.solve(Qt[dimx:, dimx:], qt[dimx:])
                lin_cost_reduce += qt[dimx:].dot(ks[t - 1])
                quad_cost_reduce += ks[t - 1] @ Qt[dimx:, dimx:] @ ks[t - 1]
                # Ks[t - 1] = -Qtinv[dimx:, dimx:] @ Qt[dimx:, :dimx]
                # ks[t - 1] = -Qtinv[dimx:, dimx:] @ qt[dimx:]
                # update Vn and vn
                Vn = Qt[:dimx, :dimx] + Qt[:dimx, dimx:] @ Ks[t - 1] + Ks[t - 1].T @ Qt[dimx:, :dimx] + Ks[t - 1].T @ Qt[dimx:, dimx:] @ Ks[t - 1]
                vn = qt[:dimx] + Qt[:dimx, dimx:] @ ks[t - 1] + Ks[t - 1].T @ (qt[dimx:] + Qt[dimx:, dimx:] @ ks[t - 1])
            # redo forward simulation and record actions...
            ls_success = False
            best_alpha = None
            best_obj = np.inf
            best_obj_estimate_reduction = None
            ks_norm = np.linalg.norm(ks)
            # print('norm of ks = ', np.linalg.norm(ks))

            # Compute rollout for all possible alphas
            alphas = np.array([ls_discount**i for i in range(ls_max_iter)])
            for i in range(ls_max_iter):
                ls_states[i,0,:] = state
            for i in range(H):
                for j, alpha in enumerate(alphas):
                    ls_ctrls[j, i, :] = alpha * ks[i] + ctrls[i] + Ks[i] @ (ls_states[j, i, :] - states[i, :])
                    if self.ubounds is not None:
                        ls_ctrls[j, i, :] = np.clip(ls_ctrls[j, i, :], self.ubounds[0], self.ubounds[1])
                ls_states[:, i + 1, :] = self.model.pred_parallel(ls_states[:, i, :], ls_ctrls[:, i, :])

            # Now do backtrack line search.
            for lsitr, ls_alpha in enumerate(alphas):
                new_states = ls_states[lsitr, :, :]
                new_ctrls = ls_ctrls[lsitr, :, :]
                new_obj = eval_obj(new_states, new_ctrls)
                expect_cost_reduction = ls_alpha * lin_cost_reduce + ls_alpha ** 2 * quad_cost_reduce / 2
                print((obj - new_obj) / (-expect_cost_reduction))
                if (obj - new_obj) / (-expect_cost_reduction) > ls_cost_threshold:
                    best_obj = new_obj
                    best_alpha = ls_alpha
                    best_alpha_idx = lsitr
                    break
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_alpha = ls_alpha
                    best_alpha_idx = lsitr
                #ls_alpha *= ls_discount
                if ks_norm < u_threshold:
                    break
            if self.verbose:
                print('line search obj %f to %f at alpha = %f' % (obj, new_obj, ls_alpha))
            if best_obj < obj or ks_norm < u_threshold:
                ls_success = True
                new_ctrls = ls_ctrls[best_alpha_idx, :, :]
                new_states = ls_states[best_alpha_idx, :, :]
                _, jxs, jus = self.model.pred_diff_parallel(new_states[:-1,:], new_ctrls)
                Jacs[:, :, :dimx] = jxs
                Jacs[:, :, dimx:] = jus
                new_obj = eval_obj(new_states, new_ctrls)
            if (not ls_success and new_obj > obj + 1e-3) or best_alpha is None:
                print('Line search fails...')
                break
            else:
                if self.verbose:
                    print('alpha is successful at %f with cost from %f to %f' % (best_alpha, obj, new_obj))
                pass
            # return since update of action is small
            if self.verbose:
                print('u update', np.linalg.norm(new_ctrls - ctrls))
            du_norm = np.linalg.norm(new_ctrls - ctrls)
            if du_norm < u_threshold:
                if self.verbose:
                    print('Break since update of control is small at %f' % (np.linalg.norm(new_ctrls - ctrls)))
                converged = True
            # ready to swap...
            states = np.copy(new_states)
            ctrls = np.copy(new_ctrls)
            obj = new_obj
            if converged:
                print('Convergence achieved within %d iterations' % itr)
                print('Cost update from %f to %f' % (initcost, obj))
                print('Final state is ', states[-1])
                break
        if not converged:
            print('ilqr fails to converge, try a new guess? Last u update is %f ks norm is %f' % (du_norm, ks_norm))
            print('ilqr is not converging...')
        return converged, states, ctrls, Ks, ks

    def run(self, info, new_obs):
        """Here I am assuming I reuse the controller for half horizon"""
        if self.reuse_feedback == 0:
            if self._guess is None:
                self._guess = np.zeros((self.horizon, self.system.ctrl_dim))
            converged, states, ctrls, Ks, ks = self.compute_ilqr(new_obs, self._guess)
            self._states = states
            self._guess = np.concatenate((ctrls[1:], np.zeros((1, self.system.ctrl_dim))), axis=0)
            return ctrls[0], None
        # Implement control logic here
        state = new_obs
        if self._need_recompute:
            converged, states, ctrls, Ks, ks = self.compute_ilqr(state, np.zeros((self.horizon, self.system.ctrl_dim)))
            self._states, self._ctrls, self._gain, self._ks = states, ctrls, Ks, ks
            self._need_recompute = False
            self._step_count = 0
            # import pdb; pdb.set_trace()
        if self._step_count == self.reuse_feedback:
            self._need_recompute = True  # recompute when last trajectory is finished... Good choice or not?
        x0, u0, k0 = self._states[self._step_count], self._ctrls[self._step_count], self._gain[self._step_count]
        u = u0 + k0 @ (state - x0)
        print('inside ilqr u0 = ', u0, 'u = ', u)
        #set_trace()
        self._step_count += 1
        return u, None
