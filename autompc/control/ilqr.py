"""
This file implement iterative LQR methods for optimization over nonlinear models.
But after realizing ilqr is different from optimization + local lqr feedback.
I start questioning myself whether this is gonna work, but anyway... It won't hurt.
"""
import numpy as np
import numpy.linalg as la
from control.matlab import dare

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from ..controller import Controller


class IterativeLQR(Controller):
    def __init__(self, system, task, model, horizon, reuse_feedback=-1):
        super().__init__(system, task, model)
        self.horizon = horizon
        self.dt = system.dt
        self._need_recompute = True  # this indicates a new iteration is required...
        self._step_count = 0
        if reuse_feedback == -1 or reuse_feedback > horizon:
            self.reuse_feedback = horizon // 2
        elif reuse_feedback is None:
            self.reuse_feedback = 1
        else:
            self.reuse_feedback = reuse_feedback
        self._guess = None

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

    def compute_ilqr(self, state, uguess, u_threshold=1e-4, ls_max_iter=5, ls_discount=0.2, ls_cost_threshold=0.3):
        """Use equations from https://medium.com/@jonathan_hui/rl-lqr-ilqr-linear-quadratic-regulator-a5de5104c750 .
        A better version is https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
        """
        Q, R, F = self.task.get_quad_cost()
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
        for itr in range(max_iter):
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
            new_states[0] = state
            ls_success = False
            ls_alpha = 1
            best_alpha = None
            best_obj = np.inf
            best_obj_estimate_reduction = None
            ks_norm = np.linalg.norm(ks)
            # print('norm of ks = ', np.linalg.norm(ks))
            for lsitr in range(ls_max_iter):
                for i in range(H):
                    new_ctrls[i] = ls_alpha * ks[i] + ctrls[i] + Ks[i] @ (new_states[i] - states[i])
                    new_states[i + 1] = self.model.pred(new_states[i], new_ctrls[i])
                new_obj = eval_obj(new_states, new_ctrls)
                expect_cost_reduction = ls_alpha * lin_cost_reduce + ls_alpha ** 2 * quad_cost_reduce / 2
                if (obj - new_obj) / (-expect_cost_reduction) > ls_cost_threshold:
                    best_obj = new_obj
                    best_alpha = ls_alpha
                    break
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_alpha = ls_alpha
                ls_alpha *= ls_discount
                if ks_norm < 1e-3:
                    break
            # print('line search obj %f to %f at alpha = %f' % (obj, new_obj, ls_alpha))
            if best_obj < obj or ks_norm < 1e-3:
                ls_success = True
                for i in range(H):
                    new_ctrls[i] = best_alpha * ks[i] + ctrls[i] + Ks[i] @ (new_states[i] - states[i])
                    new_states[i + 1], jx, ju = self.model.pred_diff(new_states[i], new_ctrls[i])
                    Jacs[i, :, :dimx] = jx
                    Jacs[i, :, dimx:] = ju
                new_obj = eval_obj(new_states, new_ctrls)
            if (not ls_success and new_obj > obj + 1e-3) or best_alpha is None:
                print('Line search fails...')
                break
            else:
                pass
                # print('alpha is successful at %f with cost from %f to %f' % (best_alpha, obj, new_obj))
            # return since update of action is small
            # print('u update', np.linalg.norm(new_ctrls - ctrls))
            if np.linalg.norm(new_ctrls - ctrls) < u_threshold:
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
            print('ilqr fails to converge, try a new guess?')
        # print('ilqr converges to trajectory', states, 'ctrls', ctrls)
        if not converged:
            print('ilqr is not converging...')
        return converged, states, ctrls, Ks, ks

    def run(self, info, new_obs):
        """Here I am assuming I reuse the controller for half horizon"""
        # if self._guess is None:
        #     self._guess = np.zeros((self.horizon, self.system.ctrl_dim))
        # converged, states, ctrls, Ks, ks = self._compute_ilqr(state, self._guess)
        # self._states = states
        # self._guess = np.concatenate((ctrls[1:], np.zeros((1, self.system.ctrl_dim))), axis=0)
        # return ctrls[0], None
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
        self._step_count += 1
        return u, None

