import autograd
import autograd.numpy as np
from pdb import set_trace

from autompc.model import Model
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

def planar_drone_dynamics(yu, g=0.0, m=1, r=0.25, I=1.0):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    x, dx, y2, dy, theta, omega, u1, u2 = yu
    return np.array([dx,
            -(u1 + u2) * np.sin(theta),
            dy,
            (u1 + u2) * np.cos(theta) - m * g,
            omega,
            r / I * (u1 - u2)])

def dt_planar_drone_dynamics(y,u,dt,g=0.0,m=1,r=0.25,I=1.0):
    yu = np.concatenate([y,u])
    #sol = solve_ivp(lambda t, y: planar_drone_dynamics(y, u, g, m, r, I), (0, dt), y, 
    #        t_eval = [dt])
    #if not sol.success:
    #    raise Exception("Integration failed due to {}".format(sol.message))
    #y = sol.y.reshape((6,))
    y += dt * np.array(planar_drone_dynamics(yu, g, m, r, I))
    return y

class PlanarDroneModel(Model):
    def __init__(self, system):
        super().__init__(system)
        self.dt = system.dt
        self.gfun = autograd.jacobian(planar_drone_dynamics)

    @staticmethod
    def get_configuration_space(system):
        cs = CS.ConfigurationSpace()
        return cs

    def traj_to_state(self, traj):
        return traj[-1].obs[:]
    
    def update_state(self, state, new_ctrl, new_obs):
        return np.copy(new_obs)

    @property
    def state_dim(self):
        return self.system.obs_dim

    def train(self, trajs):
        pass

    def pred(self, state, ctrl):
        xpred = dt_planar_drone_dynamics(state, ctrl, self.dt)
        return xpred

    def pred_diff(self, state, ctrl):
        xpred = dt_planar_drone_dynamics(state, ctrl, self.dt)
        x = np.concatenate([state, ctrl])
        jac = self.gfun(x)
        # so I get system dynamics
        A = jac[:, :6]
        B = jac[:, 6:]
        # get discrete one
        Adt = np.eye(6) + A * self.dt
        Bdt = B * self.dt

        return xpred, Adt, Bdt


    def get_parameters(self):
        return dict()

    def set_parameters(self, params):
        pass
