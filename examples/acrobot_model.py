import autograd
import autograd.numpy as np
from pdb import set_trace

from autompc.model import Model
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import ConfigSpace.conditions as CSC

def acrobot_dynamics(yu,m1=1,m2=1,l1=1,lc1=0.5,lc2=0.5,I1=1,I2=1,g=9.8):
    y = yu[:-1]
    u = yu[-1]
    cos = np.cos
    sin = np.sin
    pi = np.pi
    theta1 = y[0]
    theta2 = y[1]
    dtheta1 = y[2]
    dtheta2 = y[3]
    d1 = m1 * lc1 ** 2 + m2 * \
        (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
    phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.)
    phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2) \
           - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)  \
        + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2) + phi2
    # the following line is consistent with the java implementation and the
    # book
    ddtheta2 = (u + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) \
        / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2])


def dt_acrobot_dynamics(y,u,dt):
    y = np.copy(y)
    y[0] += np.pi
    yu = np.concatenate([y,u])
    y += dt * acrobot_dynamics(yu)
    y[0] -= np.pi
    return y


class AcrobotModel(Model):
    def __init__(self, system):
        super().__init__(system)
        self.dt = system.dt
        self.gfun = autograd.jacobian(acrobot_dynamics)

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
        xpred = dt_acrobot_dynamics(state, ctrl, self.dt)
        return xpred

    def pred_diff(self, state, ctrl):
        xpred = dt_acrobot_dynamics(state, ctrl, self.dt)
        x = np.concatenate([state, ctrl])
        #x[0] += np.pi
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
