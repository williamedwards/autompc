"""Learn a linear model for cartpole and try lqr on it.
This has been tried, but the bad result may be caused by the improper linear regression usage.
"""
import numpy as np
from scipy.integrate import solve_ivp
from joblib import Memory
from control.matlab import dare
import matplotlib.pyplot as plt

import autompc as ampc
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
memory = Memory("cache")


def cartpole_dynamics(y, u, g = 9.8, m_c = 1, m_p = 1, L = 1, b = 1.0):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    theta, omega, x, dx = y
    #return [omega,
    #        g * np.sin(theta)/L - b * omega / (m*L**2) + u * np.sin(theta)/L,
    #        dx,
    #        u]
    return [omega,
            1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                - (m_c+m_p+m_p)*g*np.sin(theta)
                - b*omega),
            dx,
            1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                (L*omega**2 + g*np.cos(theta)))]

def dt_cartpole_dynamics(y,u,dt,g=9.8,m=1,L=1,b=1.0):
    y[0] += np.pi
    sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    yo = sol.y.reshape((4,))
    yo[0] -= np.pi
    y[0] -= np.pi
    return yo


dt = 0.01

umin = -2.0
umax = 2.0

# Generate trajectories for training
num_trajs = 500

# this function collects some trajectories ready to be used for learning...
# it starts from somewhere close to the origin and apply random control for a short period
@memory.cache
def gen_trajs(num_trajs=num_trajs):
    rng = np.random.default_rng(49)
    trajs = []
    for _ in range(num_trajs):
        theta0 = rng.uniform(-0.002, 0.002, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        step = 10
        traj = ampc.zeros(cartpole, step)
        for i in range(step):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 1)
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs


def collect_data_and_learn(realA, realB):
    """In this function, I do random simulation to collect data to learn a model"""
    trajs = gen_trajs()
    # assemble data...
    next_states = []
    cur_states = []
    cur_ctrls = []
    idx0 = 0
    for traj in trajs:
        # end_ctrl_index = np.where(traj.ctrls[:, 0] == 0)[0][0]  # up to this index we have correct results...
        end_ctrl_index = traj.obs.shape[0] - 1
        next_states.append(traj.obs[1 + idx0: end_ctrl_index + 1])
        cur_states.append(traj.obs[idx0: end_ctrl_index])
        cur_ctrls.append(traj.ctrls[idx0: end_ctrl_index])
    # concatenate them...
    next_states = np.concatenate(next_states, axis=0)
    cur_states = np.concatenate(cur_states, axis=0)
    diff_states = next_states - cur_states
    next_states = diff_states
    cur_ctrls = np.concatenate(cur_ctrls, axis=0)
    features = np.concatenate((cur_states, cur_ctrls), axis=1)
    # ready for linear regression, do it state-wise, some L1 regularization may be necessary...
    # I just do Standarization without mean
    scalerx = StandardScaler(with_mean=False)
    scalerx.fit(features)
    scaled_x = scalerx.transform(features)
    scalery = StandardScaler(with_mean=False)
    scalery.fit(next_states)
    scaled_y = scalery.transform(next_states)
    # fig, ax = plt.subplots(2, 2)
    # ax = ax.reshape(-1)
    # for i in range(4):
    #     ax[i].hist(cur_states[:, i])
    # plt.show()
    # now ready to do it for each component, now no training-test set is split
    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    lr = LinearRegression(fit_intercept=False)
    # lr = Lasso(alpha=0.05, fit_intercept=False)
    lr.fit(scaled_x, scaled_y)
    v = ((scaled_y - scaled_y.mean()) ** 2).sum()
    u = ((scaled_y - lr.predict(scaled_x)) ** 2).sum()
    print('u = ', u, 'v = ', v, 'score = ', 1 - u / v)
    print('lr score is ', lr.score(scaled_x, scaled_y))
    error = lr.predict(scaled_x) - scaled_y
    error *= scalery.scale_[None, :]
    # compute error from the true dynamics (sort of...)
    true_predict = cur_states.dot(realA.T) + cur_ctrls.dot(realB.T)
    true_error = true_predict - next_states
    print('errors = ', np.mean(error, axis=0), np.mean(true_error, axis=0))
    # wow the score is so shitty... What is happening...
    # fig, ax = plt.subplots(2, 2)
    # ax = ax.reshape(-1)
    # for i in range(4):
    #     ax[i].hist([error[:, i], true_error[:, i]])
    # plt.show()
    A = lr.coef_[:, :4] / scalerx.scale_[:4][None, :] * scalery.scale_[:, None]
    B = lr.coef_[:, 4:] / scalerx.scale_[4:][None, :] * scalery.scale_[:, None]
    return A, B


if __name__ == '__main__':
    # first step is to do sysid to get the linear model
    numpy = np
    dta = np.load('disc_cartpole.npz')
    realA, realB = dta['A'], dta['B']
    A, B = collect_data_and_learn(realA * dt, realB * dt)
    # check how this feedback works for the real system
    print('realA = ', realA, '\nA = ', A / dt)
    print('realB = ', realB, '\nB = ', B / dt)
    # then lqr controller is designed
    # compute optimal gain
    Q = numpy.diag([10, 1, 10, 1])
    R = numpy.eye(1)
    useA = A + np.eye(4)  # since I'm fitting the diff
    (X, L, G) = dare(useA, B, Q, R) 
    print('eigen value ', L)
    print("gain ", G)
    realA = realA * dt + np.eye(4)
    realB = realB * dt
    fb_mat = realA - realB.dot(G[0])
    print('fb_mat = ', fb_mat)
    print('eig of fb_mat', np.linalg.eig(fb_mat)[0])
    # OK, now I do simulation, use control law u=-G^T X
    x = numpy.random.random(4) * 0.3
    states = [x]
    for i in range(500):  # simulation steps...
        x = states[-1]
        u = -np.array(G)[0].dot(x)
        new_state = dt_cartpole_dynamics(x, u, dt, g=9.8,m=1,L=1,b=1.0)
        states.append(new_state)

    states = numpy.array(states)
    fig, ax = plt.subplots(4)
    ax = ax.reshape(-1)
    for i in range(4):
        ax[i].plot(states[:, i])
    plt.show()