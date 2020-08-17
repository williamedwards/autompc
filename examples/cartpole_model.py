import autograd
import autograd.numpy as np
from pdb import set_trace

from autompc.model import Model
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

def cartpole_dynamics(yu, g = 9.8, m_c = 1, m_p = 1, L = 1, b = 1.0):
    """
    Parameters
    ----------
        y : states
        u : control
    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    theta, omega, x, dx, u = yu
    return np.array([omega,
            1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                - (m_c+m_p+m_p)*g*np.sin(theta)
                - b*omega),
            dx,
            1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                (L*omega**2 + g*np.cos(theta)))])


def dt_cartpole_dynamics(y, u, dt, g=9.8,m=1,L=1,b=1.0):
    y[0] += np.pi
    yu = np.concatenate((y, u))
    dstate = cartpole_dynamics(yu, g=g)
    y[0] -= np.pi  # recover it...
    sol = y + dstate * dt
    return sol
#def cartpole_simp_dynamics(yu, g = 9.8, m = 1, L = 1, b = 0.1):
#    """
#    Parameters
#    ----------
#        y : states
#        u : control
#
#    Returns
#    -------
#        A list describing the dynamics of the cart cart pole
#    """
#    theta, omega, x, dx,u = yu
#    return np.array([omega,
#            g * np.sin(theta)/L - b * omega / (m*L**2) + u * np.cos(theta)/L,
#            dx,
#            u])
#
#def dt_cartpole_dynamics(y,u,dt,g=9.8,m=1,L=1,b=1.0):
#    yu = np.concatenate([y,u])
#    y += dt * cartpole_simp_dynamics(yu,g,m,L,b)
#    return y


class CartpoleModel(Model):
    def __init__(self, system):
        super().__init__(system)
        self.dt = system.dt
        self.gfun = autograd.jacobian(cartpole_dynamics)

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
        xpred = dt_cartpole_dynamics(state, ctrl, self.dt)
        return xpred

    def pred_diff(self, state, ctrl):
        xpred = dt_cartpole_dynamics(state, ctrl, self.dt)
        x = np.concatenate([state, ctrl])
        x[0] += np.pi
        jac = self.gfun(x)
        # so I get system dynamics
        A = jac[:, :4]
        B = jac[:, 4:]
        # get discrete one
        Adt = np.eye(4) + A * self.dt
        Bdt = B * self.dt

        return xpred, Adt, Bdt


    def get_parameters(self):
        return dict()

    def set_parameters(self, params):
        pass
