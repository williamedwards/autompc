
# Standard libary includes
import copy 
import time

# External library includes
import numpy as np
import numpy.linalg as la
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition

# Internal libary includes
from .optimizer import Optimizer
from ..trajectory import Trajectory
from ..utils.cs_utils import get_hyper_bool, get_hyper_int

def inverse_semidefinite(A, damping=1e-3):
    try:
        w,V = np.linalg.eigh(A)
    except:
        breakpoint()
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
    - **max_iter** *(Type: int, Low: 10, Upper: 50, Default: 20)*: Maximum number of iterations
        to run iLQR at each step.    
    - **random_restarts** *(Optional, Type: bool, Default: False)*: When true, random restarts
        are run periodically.  Hyperparameter only present when enable_random_restarts=True.
    - **random_restart_frequency** *(Optional, Type: bool, Lower: 1, Upper: 10, Default: 3)*:
        Try random restart every `random_restart_frequency` steps.  Enabled only when
        `random_restarts` is true. Hyperparameter only present when enable_random_restarts=True.
    """
    def __init__(self, system, verbose=False, enable_random_restarts=False):
        self.verbose = verbose
        self.enable_random_restarts=enable_random_restarts
        super().__init__(system, "IterativeLQR")

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

        if self.enable_random_restarts:
            random_restarts = CategoricalHyperparameter(
                name="random_restarts",
                choices=["true", "false"],
                default_value="false")
            random_restart_frequency = UniformIntegerHyperparameter(
                name="random_restart_frequency",
                lower=1, upper=10, default_value=3)
            use_random_restarts = InCondition(
                child=random_restart_frequency,
                parent=random_restarts,
                values=["true"])
            cs.add_hyperparameters([random_restarts, random_restart_frequency])
            cs.add_condition(use_random_restarts)
        return cs

    def set_config(self, config):
        self.horizon = get_hyper_int(config, "horizon")
        self.max_iter = get_hyper_int(config, "max_iter")
        self.frequency = get_hyper_int(config, "frequency")
        if self.frequency >= self.horizon:
            self.frequency = self.horizon - 1
        if self.enable_random_restarts:
            self.random_restarts = get_hyper_bool(config, "random_restarts")
            self.random_restart_frequency = get_hyper_int(config, "random_restart_frequency")

    def reset(self, seed=0):
        self._guess = None
        self._traj = None
        self._gains = None
        self._step = 0
        self._steps_since_random_restart = 0
        self._rng = np.random.default_rng(seed)

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

    def compute_ilqr(self, x0, uguess, u_threshold=1e-3,
        ls_max_iter=10, ls_discount=0.2, ls_cost_threshold=0.3):
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
                # first assemble Ct and ct, they are linearized at current state
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
                K = -np.linalg.solve(Quu, Qux)
                k = -np.linalg.solve(Quu, qu)
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
            
        if self.verbose and not converged :
            print('ilqr fails to converge, try a new guess? Last u update is %f ks norm is %f' % (du_norm, ks_norm))
            print('ilqr is not converging...')
        return converged, states, ctrls, Ks, obj
    
    def set_guess(self, guess : Trajectory) -> None:
        assert len(guess) == self.horizon,"Invalid guess provided"
        self._guess = guess.ctrls

    def step(self, obs):
        substep = self._step % self.frequency
        self._step += 1
        if self._guess is None:
            self._guess = np.zeros((self.horizon, self.system.ctrl_dim))
        if substep == 0: 
            converged, states, ctrls, Ks, obj = self.compute_ilqr(obs, self._guess)

            #Perform random restart
            if self.enable_random_restarts and self.random_restarts:
                self._steps_since_random_restart += 1
                if self._steps_since_random_restart >= self.random_restart_frequency:
                    print("Running random restart!")
                    random_guess = self._rng.uniform(
                        low=self.ubounds[0], 
                        high=self.ubounds[1], 
                        size=(self.horizon, self.system.ctrl_dim))
                    _, rstates, rctrls, rKs, robj = self.compute_ilqr(obs, random_guess)
                    print(f"Original Objective: {obj}\t Random Objective: {robj}")
                    if robj < obj:
                        obj = robj
                        states = rstates
                        ctrls = rctrls
                        Ks = rKs
                    self._steps_since_random_restart = 0

            self._guess = np.concatenate((ctrls[1:], ctrls[-1:]*0), axis=0)
            self._traj = Trajectory(self.model.state_system, states, 
                np.vstack([ctrls, np.zeros(self.system.ctrl_dim)]))
            self._gains = Ks
            return ctrls[0]
        else:
            #just reuse prior strajectory and gains
            return self._traj.ctrls[substep]+self._gains[substep]@(obs-self._traj.obs[substep])

    def get_traj(self):
        return self._traj.clone()
