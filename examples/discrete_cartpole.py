import autograd
import autograd.numpy as np
import numpy
from control.matlab import dare
import matplotlib.pyplot as plt


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
    yu = numpy.concatenate((y, [u]))
    dstate = cartpole_dynamics(yu, g=g)
    y[0] -= np.pi  # recover it...
    sol = y + dstate * dt
    return sol


gfun = autograd.jacobian(cartpole_dynamics)
x0 = numpy.array([np.pi, 0, 0, 0, 0])
jac = gfun(x0)
# so I get system dynamics
A = jac[:, :4]
B = jac[:, 4:]
# get discrete one
dt = 0.025
Adt = numpy.eye(4) + A * dt
Bdt = B * dt
# compute optimal gain
Q = numpy.diag([10, 1, 10, 1])
R = numpy.eye(1)
(X, L, G) = dare(Adt, Bdt, Q, R) 
print('eigen value ', L)
print("gain ", G)
# OK, now I do simulation, use control law u=-G^T X
x = numpy.random.random(4) * 0.3
print(x)
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