def cartPole_dynamics(y, u, g = 9.8, m = 1, L = 1, b = 0.1):
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
    return [omega,
            g * np.sin(theta)/L - b * omega / (m*L**2) + u * np.sin(theta)/L,
            dx,
            u]

def difDrive_dynamics(states, control, r = 0.036, L = 0.258):
    """
    Parameters
    ----------
        states : The world-frame x-y coordinates and orientation of the
                 differential drive system
        control : The turning velocity of each wheel of the differential drive
        r : The radius of each wheel
        L : The distance between the wheels

    Returns
    -------
        A list containing the dynamics of the differential drive system
    """
    x, y, theta = states
    uR, uL = control
    return  r * [np.cos(theta) * uR + np.cos(theta) * uL,
                 np.sin(theta) * uR + np.sin(theta) * uL,
                 (uR - uL)/ L]

def acrobot_dynamics(states, control, m1 = 1, m2 = 0.5, L1 = 1, L2 = 2, g = 9.81):
    """
    Parameters
    ----------
        states : The two angles and angular velocities of the acrobot
        control : The 2-dimensional control (applied only at the joint of the second angle)

    Returns
    -------
        A list with the 4-dimensional dynamics of the acrobot
    """


    theta1, theta2, dtheta1, dtheta2 = states
    u = control
    alpha1 = (m1 + m2) * L1**2
    alpha2 = m2 * L2**2
    alpha3 = m2 * L1 * L2
    beta1 = (m1 + m2) * g * L1
    beta2 = m2 * g * L2

    m11 = alpha1 + alpha2 + 2 * alpha3 * np.cos(theta2)
    m12 = alpha3 * np.cos(theta2) + alpha2
    m21 = m12
    m22 = alpha2

    h1 = - 2 * alpha3 * dtheta1 * dtheta2 * np.sin(theta2) - alpha3 * dtheta2**2 * np.sin(theta2)
    h2 = alpha3 * dtheta1**2 * np.sin(theta2)

    g1 = -beta1 * np.sin(theta1) - beta2 * np.sin(theta1 + theta2)
    g2 = -beta2 * np.sin(theta1 + theta2)

    M = np.array([[m11, m12], [m21, m22]])
    H = np.array([[h1], [h2]])
    G = np.array([[g1], [g2]])
    B = np.array([[0], [1]])

    ddthetas = - np.dot(scipy.linalg.pinv2(M), ( (H + G) - B * u))
    return np.vstack((np.array([[dtheta1], [dtheta2]]), ddthetas))

import scipy.linalg
import numpy as np

temp = acrobot_dynamics([0,1,2,3], np.array([[0],[1]]))
