
# Standard libary includes
import copy 

# External library includes
import numpy as np
import numpy.linalg as la
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

# Internal libary includes
from .optimizer import Optimizer

class IterativeLQR(Optimizer):
    """
    Iterative Linear Quadratic Regulator (ILQR) can be considered as a Dynamic Programming (DP) method to solve trajectory optimization problems.
    Usually DP requires discretization and scales exponentially to the dimensionality of state and control so it is not practical beyond simple problems.
    However, if DP is applied locally around some nominal trajectory, the value function and policy function can be expressed in closed form, thus avoiding exponential complexity.
    Specifically, starting from some nominal trajectory, the dynamics equation is linearized and the cost function is approximated by a quadratic function.
    In this way, the control policy is the nominal control plus linear feedback of state deviation.
    This policy is applied to the system and a updated nominal trajectory is obtained.
    By doing this process iteratively, the optimal trajectory can be obtained.
    Some reference to understand this technique can be found at `this blog <https://jonathan-hui.medium.com/rl-lqr-ilqr-linear-quadratic-regulator-a5de5104c750>`_, `this blog <https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/>`_, and `this slide <https://katefvision.github.io/katefSlides/RECITATIONtrajectoryoptimization_katef.pdf>`_. 

    Hyperparameters:

    - **horizon** *(Type: int, Low: 5, Upper: 25, Default: 20)*: MPC Optimization Horizon.
    """
    def __init__(self, system, verbose=False):
        super().__init__(system, "IterativeLQR")
        self.verbose = verbose

    def get_default_config_space(self):
        cs = ConfigurationSpace()
        horizon = UniformIntegerHyperparameter(name="horizon",
                lower=5, upper=25, default_value=20)
        cs.add_hyperparameter(horizon)
        return cs

    def set_config(self, config):
        self.horizon = config["horizon"]

    def reset(self):
        self._guess = None

    def set_ocp(self, ocp):
        super().set_ocp(ocp)
        if ocp.are_ctrl_bounded:
            bounds = ocp.get_ctrl_bounds()
            self.ubounds = (bounds[:,0], bounds[:,1])
        else:
            self.ubounds = None

    def get_state(self):
        return {"guess" : copy.deepcopy(self._guess) }

    def is_compatible(self, model, ocp):
        return (model.is_diff
                and ocp.get_cost().is_twice_diff
                and not ocp.are_obs_bounded)

    def set_state(self, state):
        self._guess = copy.deepcopy(state["guess"])

    def compute_ilqr(self, state, uguess, u_threshold=1e-3, max_iter=50, 
            ls_max_iter=10, ls_discount=0.2, ls_cost_threshold=0.3, silent=False):
        """Use equations from https://medium.com/@jonathan_hui/rl-lqr-ilqr-linear-quadratic-regulator-a5de5104c750 .
        A better version is https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
        Here I do not have Hessian correction since I'm certain all my matrices are SPD
        """
        cost = self.ocp.get_cost()
        H = self.horizon
        dt = self.system.dt

        def eval_obj(xs, us):
            obj = 0
            for i in range(H):
                obj += dt * (cost.eval_obs_cost(xs[i, :self.system.obs_dim]) + cost.eval_ctrl_cost(us[i]))
            obj += cost.eval_term_obs_cost(xs[-1, :self.system.obs_dim])
            return obj

        dimx, dimu = self.model.state_dim, self.system.ctrl_dim
        obsdim = self.system.obs_dim
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
        Ct = np.zeros((dimx + dimu, dimx + dimu))
        ct = np.zeros(dimx + dimu)
        converged = False
        du_norm = np.inf
        for itr in range(max_iter):
            if self.verbose:
                print('At iteration %d' % itr)
            # compute at the last step, Vn and vn, just hessian and gradient at the last state
            _, cost_jac, cost_hess = cost.eval_term_obs_cost_hess(states[H, :obsdim])
            Vn = np.zeros((dimx, dimx))
            vn = np.zeros(dimx)
            Vn[:obsdim, :obsdim] = cost_hess
            vn[:obsdim] = cost_jac
            lin_cost_reduce = quad_cost_reduce = 0
            for t in range(H, 0, -1):  # so run for H steps...
                # first assemble Ct and ct, they are linearized at current state
                Q  = np.zeros((dimx, dimx))
                Qx = np.zeros(dimx)
                _, Qx[:obsdim], Q[:obsdim, :obsdim] = cost.eval_obs_cost_hess(states[t - 1, :obsdim])
                _, Ru, R = cost.eval_ctrl_cost_hess(ctrls[t - 1])
                Ct[:dimx, :dimx] = Q * dt
                Ct[dimx:, dimx:] = R * dt
                ct[:dimx] = Qx * dt
                ct[dimx:] = Ru * dt
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
                ls_states[:, i + 1, :] = self.model.pred_batch(ls_states[:, i, :], ls_ctrls[:, i, :])

            # Now do backtrack line search.
            for lsitr, ls_alpha in enumerate(alphas):
                new_states = ls_states[lsitr, :, :]
                new_ctrls = ls_ctrls[lsitr, :, :]
                new_obj = eval_obj(new_states, new_ctrls)
                expect_cost_reduction = ls_alpha * lin_cost_reduce + ls_alpha ** 2 * quad_cost_reduce / 2
                #print((obj - new_obj) / (-expect_cost_reduction))
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
                _, jxs, jus = self.model.pred_diff_batch(new_states[:-1,:], new_ctrls)
                Jacs[:, :, :dimx] = jxs
                Jacs[:, :, dimx:] = jus
                new_obj = eval_obj(new_states, new_ctrls)
            if (not ls_success and new_obj > obj + 1e-3) or best_alpha is None:
                if not silent:
                    print('Line search fails...')
                break
            else:
                if self.verbose and not silent:
                    print('alpha is successful at %f with cost from %f to %f' % (best_alpha, obj, new_obj))
                pass
            # return since update of action is small
            if self.verbose and not silent:
                print('u update', np.linalg.norm(new_ctrls - ctrls))
            du_norm = np.linalg.norm(new_ctrls - ctrls)
            if du_norm < u_threshold:
                if self.verbose and not silent:
                    print('Break since update of control is small at %f' % (np.linalg.norm(new_ctrls - ctrls)))
                converged = True
            # ready to swap...
            states = np.copy(new_states)
            ctrls = np.copy(new_ctrls)
            obj = new_obj
            if converged:
                if not silent:
                    print('Convergence achieved within %d iterations' % itr)
                    print('Cost update from %f to %f' % (initcost, obj))
                    print('Final state is ', states[-1])
                break
        if not converged and not silent:
            print('ilqr fails to converge, try a new guess? Last u update is %f ks norm is %f' % (du_norm, ks_norm))
            print('ilqr is not converging...')
        return converged, states, ctrls, Ks, ks

    def step(self, state, silent=True):
        if self._guess is None:
            self._guess = np.zeros((self.horizon, self.system.ctrl_dim))
        converged, states, ctrls, Ks, ks = self.compute_ilqr(state, self._guess,
                silent=silent)
        self._guess = np.concatenate((ctrls[1:], np.zeros((1, self.system.ctrl_dim))), axis=0)
        return ctrls[0]
