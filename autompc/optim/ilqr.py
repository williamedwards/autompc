
# Standard libary includes
import copy 
import time

# External library includes
import numpy as np
import numpy.linalg as la
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

# Internal libary includes
from .optimizer import Optimizer
from ..trajectory import Trajectory

def inverse_semidefinite(A, damping=1e-3):
    w,V = np.linalg.eigh(A)
    winv = np.divide(1.0,(w + damping))
    return np.multiply(V.T,winv) @ V
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

    We also add a hyperparameter that allows only recomputing the optimal
    trajectory every `frequency` timesteps.  This reuses the prior trajectory
    and gains for the other `frequency-1` timesteps.

    Hyperparameters:

    - **horizon** *(Type: int, Low: 5, Upper: 25, Default: 20)*: MPC optimization horizon.
    - **frequency** *(Type: int, Low: 1, Upper: 5, Default: 1)*: Recompute trajectory every
      `frequency` steps.  Otherwise, will use the prior gains.
    """
    def __init__(self, system, verbose=False):
        super().__init__(system, "IterativeLQR")
        self.verbose = verbose

    def get_default_config_space(self):
        cs = ConfigurationSpace()
        horizon = UniformIntegerHyperparameter(name="horizon",
                lower=5, upper=25, default_value=20)
        frequency = UniformIntegerHyperparameter(name="frequency",
                lower=1, upper=5, default_value=1)
        max_iter = UniformIntegerHyperparameter(name="max_iter",
                lower=10, upper=50, default_value=20)
        cs.add_hyperparameter(horizon)
        cs.add_hyperparameter(max_iter)
        cs.add_hyperparameter(frequency)
        return cs

    def set_config(self, config):
        self.horizon = config["horizon"]
        self.max_iter = config["max_iter"]
        self.frequency = config["frequency"]
        if self.frequency >= self.horizon:
            self.frequency = self.horizon - 1

    def reset(self):
        self._guess = None
        self._traj = None
        self._gains = None
        self._step = 0

    def set_ocp(self, ocp):
        super().set_ocp(ocp)
        if ocp.are_ctrl_bounded:
            bounds = ocp.get_ctrl_bounds()
            self.ubounds = (bounds[:,0], bounds[:,1])
        else:
            self.ubounds = None

    def get_state(self):
        return {"guess" : copy.deepcopy(self._guess) }

    def model_requirements(self):
        return {'is_diff':True}
    
    def ocp_requirements(self):
        return {'are_obs_bounded':False}

    def cost_requirements(self):
        return {'is_twice_diff':True}

    def set_state(self, state):
        self._guess = copy.deepcopy(state["guess"])

    '''def compute_ilqr(self, x0, uguess, u_threshold=1e-3,
        ls_max_iter=10, ls_discount=0.2, ls_cost_threshold=0.3, timeout=np.inf):
        """Use equations from https://medium.com/@jonathan_hui/rl-lqr-ilqr-linear-quadratic-regulator-a5de5104c750 .
        A better version is https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
        """
        cost = self.ocp.get_cost()
        H = self.horizon
        dt = self.system.dt

        def eval_obj(xs, us):
            obj = 0
            for i in range(H):
                obj += dt * cost.incremental(xs[i, :self.system.obs_dim],us[i])
            obj += cost.terminal(xs[-1, :self.system.obs_dim])
            return obj

        dimx, dimu = self.model.state_dim, self.system.ctrl_dim
        obsdim = self.system.obs_dim
        # handy variables...
        states, new_states = np.zeros((2, H + 1, dimx))
        ctrls, new_ctrls = np.zeros((2, H, dimu))
        #ls_states = np.zeros((ls_max_iter, H + 1, dimx))
        #ls_ctrls = np.zeros((ls_max_iter, H, dimu))
        Ks = np.zeros((H, dimu, dimx))
        ks = np.zeros((H, dimu))
        Jacs = np.zeros((H, dimx, dimx + dimu))  # Jacobian from dynamics...
        # first forward simulation
        states[0] = x0
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
        Q  = np.zeros((dimx, dimx))
        Qx = np.zeros(dimx)        
        converged = False
        du_norm = np.inf
        ls_alpha = 1.0
        start = time.time()
        for itr in range(self.max_iter):
            if self.verbose:
                print('At iteration %d' % itr)
            # compute at the last step, Vn and vn, just hessian and gradient at the last state
            _, cost_jac, cost_hess = cost.terminal_hess(states[H, :obsdim])
            Vn = np.zeros((dimx, dimx))
            vn = np.zeros(dimx)
            Vn[:obsdim, :obsdim] = cost_hess
            vn[:obsdim] = cost_jac
            lin_cost_reduce = quad_cost_reduce = 0
            for t in range(H, 0, -1):  # so run for H steps...
                # first assemble Ct and ct, they are linearized at current state\                
                _, Qx[:obsdim], Ru, Q[:obsdim, :obsdim], _, R = cost.incremental_hess(states[t - 1, :obsdim],ctrls[t - 1])
                Ct[:dimx, :dimx] = Q * dt
                Ct[dimx:, dimx:] = R * dt
                ct[:dimx] = Qx * dt
                ct[dimx:] = Ru * dt
                Qt = Ct + Jacs[t - 1].T @ Vn @ Jacs[t - 1]
                qt = ct + Jacs[t - 1].T @ (vn)  # here Vn @ states[t] may be necessary
                # ready to compute feedback
                Qxx = Qt[:dimx, :dimx]
                Quu = Qt[dimx:, dimx:]
                Qux = Qt[dimx:, :dimx]
                qx = qt[:dimx]
                qu = qt[dimx:]
                if self.verbose:
                    print(cost)
                    print(cost.incremental_hess(states[t - 1, :obsdim],ctrls[t - 1]))
                #QuuInv = inverse_semidefinite(Quu,1e-2)
                #K = -QuuInv @ Qux
                #k = -QuuInv @ qu
                K = -np.linalg.solve(Quu,Qux)
                k = -np.linalg.solve(Quu,qu)
                lin_cost_reduce += qu.dot(k)
                quad_cost_reduce += k @ Quu @ k
                # update Vn and vn
                Vn = Qxx + Qux.T @ K + K.T @ Qux + K.T @ Quu @ K
                vn = qx + Qux.T @ k + K.T @ (qu + Quu @ k)
                Ks[t - 1] = K
                ks[t - 1] = k
            ks_norm = np.linalg.norm(ks)
            # print('norm of ks = ', ks_norm)

            # Now do backtrack line search.
            best_alpha = None
            best_obj = np.inf
            
            alpha0 = ls_alpha
            du = []
            new_states[0] = x0
            for i in range(H):
                du.append(ks[i] + Ks[i] @ (new_states[i, :] - states[i, :]))
                new_states[i+1] = self.model.pred(new_states[i],ctrls[i] + du[-1])
            for lsitr in range(ls_max_iter):
                new_states[0] = x0
                for i in range(H):
                    #new_ctrls[i] = ctrls[i] + ls_alpha* ks[i] + Ks[i] @ (new_states[i, :] - states[i, :])
                    new_ctrls[i] = ctrls[i] + ls_alpha* du[i]
                    if self.ubounds is not None:
                        new_ctrls[i] = np.clip(new_ctrls[i], self.ubounds[0], self.ubounds[1])
                    new_states[i+1] = self.model.pred(new_states[i],new_ctrls[i])
    
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
                if ks_norm*ls_alpha < u_threshold:
                    break
                ls_alpha *= ls_discount
            
            #need to resimulate for the best alpha
            ls_alpha = best_alpha
            new_states[0] = x0
            for i in range(H):
                #new_ctrls[i] = ctrls[i] + ls_alpha* ks[i] + Ks[i] @ (new_states[i, :] - states[i, :])
                new_ctrls[i] = ctrls[i] + ls_alpha* du[i]
                if self.ubounds is not None:
                    new_ctrls[i] = np.clip(new_ctrls[i], self.ubounds[0], self.ubounds[1])
                new_states[i+1] = self.model.pred(new_states[i],new_ctrls[i])
            
            if self.verbose :
                print('line search obj %f to %f at alpha = %f' % (obj, best_obj, best_alpha))
            if ks_norm*ls_alpha < u_threshold:
                if self.verbose :
                    print('Break since update of control is small at %f' % (np.linalg.norm(new_ctrls - ctrls)))
                converged = True
            elif best_obj < obj:
                _, jxs, jus = self.model.pred_diff_batch(new_states[:-1,:], new_ctrls)
                Jacs[:, :, :dimx] = jxs
                Jacs[:, :, dimx:] = jus
                new_obj = eval_obj(new_states, new_ctrls)
                if ks_norm < u_threshold:
                    print("Stopped because u delta is low")
                    best_alpha = 1.0
            else:
                if self.verbose :
                    print("Line search fails, best obj",best_obj,"obj",obj,"best_alpha",best_alpha)
                break
            
            # return since update of action is small
            if self.verbose :
                print('u update', np.linalg.norm(new_ctrls - ctrls))
            du_norm = np.linalg.norm(new_ctrls - ctrls)
            if du_norm < u_threshold:
                if self.verbose :
                    print('Break since update of control is small at %f' % (np.linalg.norm(new_ctrls - ctrls)))
                converged = True
            
            #update alpha guess
            if best_alpha == alpha0:  #cost reduction on step 1, increase step size
                ls_alpha = min(1,ls_alpha/(1.0-ls_discount))

            # ready to swap...
            states[:] = new_states
            ctrls[:] = new_ctrls
            obj = new_obj
            if converged:
                if self.verbose :
                    print('Convergence achieved within %d iterations' % itr)
                    print('Cost update from %f to %f' % (initcost, obj))
                    print('Final state is ', states[-1])
                break
            if time.time()-start > timeout:
                break
            
        if self.verbose and not converged :
            print('ilqr fails to converge, try a new guess? Last u update is %f ks norm is %f' % (du_norm, ks_norm))
            print('ilqr is not converging...')
        return converged, states, ctrls, Ks
    '''
    
    def compute_ilqr(self, x0, uguess, u_threshold=1e-3, ls_max_iter=10, ls_discount=0.2, ls_cost_threshold=0.3, timeout=np.inf, silent=True):
        """Use equations from https://medium.com/@jonathan_hui/rl-lqr-ilqr-linear-quadratic-regulator-a5de5104c750 .
        A better version is https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
        Here I do not have Hessian correction since I'm certain all my matrices are SPD
        """
        start = time.time()
        cost = self.ocp.get_cost()
        H = self.horizon
        dt = self.system.dt

        def eval_obj(xs, us):
            obj = 0
            for i in range(H):
                obj += dt * cost.incremental(xs[i, :self.system.obs_dim],us[i])
            obj += cost.terminal(xs[-1, :self.system.obs_dim])
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
        states[0] = x0
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
        for itr in range(self.max_iter):
            if time.time()-start > timeout:
                if self.verbose:
                    print('Timeout')
                break
            if self.verbose:
                print('At iteration %d' % itr)
            # compute at the last step, Vn and vn, just hessian and gradient at the last state
            _, cost_jac, cost_hess = cost.terminal_hess(states[H, :obsdim])
            Vn = np.zeros((dimx, dimx))
            vn = np.zeros(dimx)
            Vn[:obsdim, :obsdim] = cost_hess
            vn[:obsdim] = cost_jac
            lin_cost_reduce = quad_cost_reduce = 0
            for t in range(H, 0, -1):  # so run for H steps...
                # first assemble Ct and ct, they are linearized at current state
                Q  = np.zeros((dimx, dimx))
                Qx = np.zeros(dimx)
                _, Qx[:obsdim], Ru, Q[:obsdim, :obsdim], _, R = cost.incremental_hess(states[t - 1, :obsdim],ctrls[t - 1])
                Ct[:dimx, :dimx] = Q * dt
                Ct[dimx:, dimx:] = R * dt
                ct[:dimx] = Qx * dt
                ct[dimx:] = Ru * dt
                Qt = Ct + Jacs[t - 1].T @ Vn @ Jacs[t - 1]
                qt = ct + Jacs[t - 1].T @ (vn)  # here Vn @ states[t] may be necessary
                # ready to compute feedback
                # Qtinv = inverse_semidefinite(Qt[dimx:, dimx:])
                Ks[t - 1] = -np.linalg.solve(Qt[dimx:, dimx:], Qt[dimx:, :dimx])
                ks[t - 1] = -np.linalg.solve(Qt[dimx:, dimx:], qt[dimx:])
                lin_cost_reduce += qt[dimx:].dot(ks[t - 1])
                quad_cost_reduce += ks[t - 1] @ Qt[dimx:, dimx:] @ ks[t - 1]
                # Ks[t - 1] = -Qtinv @ Qt[dimx:, :dimx]
                # ks[t - 1] = -Qtinv @ qt[dimx:]
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
            #print('Backward pass', duration)
            # Compute rollout for all possible alphas
            alphas = np.array([ls_discount**i for i in range(ls_max_iter)])
            for i in range(ls_max_iter):
                ls_states[i,0,:] = x0
            for i in range(H):
                for j, alpha in enumerate(alphas):
                    ls_ctrls[j, i, :] = alpha * ks[i] + ctrls[i] + Ks[i] @ (ls_states[j, i, :] - states[i, :])
                if self.ubounds is not None:
                    ls_ctrls[:, i, :] = np.clip(ls_ctrls[:, i, :], self.ubounds[0], self.ubounds[1])
                ls_states[:, i + 1, :] = self.model.pred_batch(ls_states[:, i, :], ls_ctrls[:, i, :])
            #print('Rollout: ', duration)

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
            #print('Line search: ', duration)
            
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
        return converged, states, ctrls, Ks
    
    def set_guess(self, guess : Trajectory) -> None:
        assert len(guess) == self.horizon,"Invalid guess provided"
        self._guess = guess.ctrls

    def step(self, obs):
        substep = self._step % self.frequency
        self._step += 1
        if self._guess is None:
            self._guess = np.zeros((self.horizon, self.system.ctrl_dim))
        if substep == 0: 
            converged, states, ctrls, Ks = self.compute_ilqr(obs, self._guess)
            self._guess = np.concatenate((ctrls[1:], ctrls[-1:]*0), axis=0)
            self._traj = Trajectory(self.model.state_system, states, 
                np.vstack([ctrls, np.zeros(self.system.ctrl_dim)]))
            self._gains = Ks
            return ctrls[0]
        else:
            #just reuse prior strajectory and gains
            return np.clip(self._traj.ctrls[substep]+self._gains[substep]@(obs-self._traj.obs[substep]),
                           self.ubounds[0], self.ubounds[1])

    def get_traj(self):
        return self._traj.clone()
