import time
from pdb import set_trace
import sys, os, io
sys.path.append(os.getcwd() + "/..")

import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import pickle
from joblib import Memory
import torch

from scipy.integrate import solve_ivp
import scipy

memory = Memory("cache")

#cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
acrobot = ampc.System(["theta1", "theta2", "dtheta1", "dtheta2"], ["u"])

#def acrobot_dynamics(states, control, m1 = 1, m2 = 0.5, L1 = 1, L2 = 2, g = 9.81):
#    """
#    Parameters
#    ----------
#        states : The two angles and angular velocities of the acrobot
#        control : The 2-dimensional control (applied only at the joint of the second angle)
#
#    Returns
#    -------
#        A list with the 4-dimensional dynamics of the acrobot
#    """
#
#
#    theta1, theta2, dtheta1, dtheta2 = states
#    u = control
#    alpha1 = (m1 + m2) * L1**2
#    alpha2 = m2 * L2**2
#    alpha3 = m2 * L1 * L2
#    beta1 = (m1 + m2) * g * L1
#    beta2 = m2 * g * L2
#
#    m11 = alpha1 + alpha2 + 2 * alpha3 * np.cos(theta2)
#    m12 = alpha3 * np.cos(theta2) + alpha2
#    m21 = m12
#    m22 = alpha2
#
#    h1 = - 2 * alpha3 * dtheta1 * dtheta2 * np.sin(theta2) - alpha3 * dtheta2**2 * np.sin(theta2)
#    h2 = alpha3 * dtheta1**2 * np.sin(theta2)
#
#    g1 = -beta1 * np.sin(theta1) - beta2 * np.sin(theta1 + theta2)
#    g2 = -beta2 * np.sin(theta1 + theta2)
#
#    M = np.array([[m11, m12], [m21, m22]])
#    H = np.array([[h1], [h2]])
#    G = np.array([[g1], [g2]])
#    B = np.array([[0], [1]])
#
#    ddthetas = - np.dot(scipy.linalg.pinv2(M), ( (H + G) - B * u))
#    return np.vstack((np.array([[dtheta1], [dtheta2]]), ddthetas))

def acrobot_dynamics(y,u,m1=1,m2=1,l1=1,lc1=0.5,lc2=0.5,I1=1,I2=1,g=9.8):
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
    y += dt * acrobot_dynamics(y,u[0])
    y[0] -= np.pi
    return y

def animate_acrobot(fig, ax, dt, traj):
    ax.grid()
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])

    line1, = ax.plot([0.0, 0.0], [0.0, -1.0], 'o-', lw=2)
    line2, = ax.plot([0.0, -1.0], [0.0, -2.0], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ctrl_text = ax.text(0.7, 0.95, '', transform=ax.transAxes)

    def init():
        line1.set_data([0.0, 0.0], [0.0, -1.0])
        line2.set_data([0.0, -1.0], [0.0, -2.0])
        time_text.set_text('')
        return line1, line2, time_text

    def animate(i):
        #i = min(i, ts.shape[0])
        x1 = -np.cos(traj[i, "theta1"]+ np.pi/2)
        y1 = np.sin(traj[i, "theta1"] + np.pi/2)
        x2 = x1 - np.cos(traj[i, "theta1"] + traj[i, "theta2"] + np.pi/2)
        y2 = y1 + np.sin(traj[i, "theta1"] + traj[i, "theta2"] + np.pi/2)
        line1.set_data([0, x1], [0, y1])
        line2.set_data([x1, x2], [y1, y2])
        time_text.set_text('t={:.2f}'.format(dt*i))
        ctrl_text.set_text("u={:.2f}".format(traj[i,"u"]))
        return line1, line2, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani

dt = 0.05
acrobot.dt = dt

umin = -20.0
umax = 20.0
udmax = 0.25

# Generate trajectories for training
num_trajs = 500

#from cartpole_model import CartpoleModel
#from autompc.control import FiniteHorizonLQR
#from autompc.sysid.dummy_linear import DummyLinear
#
#def get_generation_controller():
#    truedyn = CartpoleModel(cartpole)
#    _, A, B = truedyn.pred_diff(np.zeros(4,), np.zeros(1))
#    model = DummyLinear(cartpole, A, B)
#    Q = np.eye(4)
#    R = 0.01 * np.eye(1)
#
#    from autompc.tasks.quad_cost import QuadCost
#    cost = QuadCost(cartpole, Q, R)
#
#    from autompc.tasks.task import Task
#
#    task = Task(cartpole)
#    task.set_cost(cost)
#    task.set_ctrl_bound("u", -20.0, 20.0)
#    cs = FiniteHorizonLQR.get_configuration_space(cartpole, task, model)
#    cfg = cs.get_default_configuration()
#    cfg["horizon"] = 1000
#    con = ampc.make_controller(cartpole, task, model, FiniteHorizonLQR, cfg)
#    return con

# Generate trajectories for training
num_trajs = 500

@memory.cache
def gen_trajs(traj_len, num_trajs=num_trajs, dt=dt, rand_contr_prob=1.0):
    rng = np.random.default_rng(49)
    trajs = []
    #con = get_generation_controller()
    for _ in range(num_trajs):
        theta0 = rng.uniform(-1.0, 1.0, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(acrobot, traj_len)
        traj.obs[:] = y
        #if rng.random() < rand_contr_prob:
        actuate = False
        #else:
        #    actuate = True
        #    constate = con.traj_to_state(traj[:1])
        for i in range(traj_len):
            traj[i].obs[:] = y
            #if u[0] > umax:
            #    u[0] = umax
            #if u[0] < umin:
            #    u[0] = umin
            #u += rng.uniform(-udmax, udmax, 1)
            if not actuate:
                u  = rng.uniform(umin, umax, 1)
            else:
                u, constate = con.run(constate, y)
            y = dt_acrobot_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs
trajs = gen_trajs(4)
trajs2 = gen_trajs(200)
#trajs3 = gen_trajs(200, rand_contr_prob = 0.5)

from autompc.sysid import (GaussianProcess, 
        LargeGaussianProcess, 
        ApproximateGaussianProcess, MLP)
from autompc.control import IterativeLQR, FiniteHorizonLQR

def create_lqr_controller(task):
    from acrobot_model import AcrobotModel
    from autompc.sysid.dummy_linear import DummyLinear
    true_dyn = AcrobotModel(acrobot)
    _, A, B = true_dyn.pred_diff(np.zeros(4,), np.zeros(1))
    model = DummyLinear(acrobot, A, B)
    
    cs = FiniteHorizonLQR.get_configuration_space(acrobot, task, model)
    cfg = cs.get_default_configuration()
    cfg["horizon"] = 1000
    con  = ampc.make_controller(acrobot, task, model, FiniteHorizonLQR, cfg)
    return con

@memory.cache
def train_approx_gp_inner(num_trajs):
    cs = ApproximateGaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, ApproximateGaussianProcess, cfg)
    model.train(trajs3[-num_trajs:])
    return model.get_parameters()

def train_approx_gp(num_trajs):
    cs = ApproximateGaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, ApproximateGaussianProcess, cfg)
    params = train_approx_gp_inner(num_trajs)
    model.set_parameters(params)
    return model

@memory.cache
def train_mlp_inner(num_trajs):
    cs = MLP.get_configuration_space(acrobot)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(acrobot, MLP, cfg)
    model.train(trajs2[-num_trajs:])
    return model.get_parameters()

def train_mlp(num_trajs):
    cs = MLP.get_configuration_space(acrobot)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(acrobot, MLP, cfg)
    params = train_mlp_inner(num_trajs)
    model.set_parameters(params)
    return model


def init_ilqr(model, task, hori=40):
    ubound = np.array([[-15], [15]])
    mode = 'auglag'
    ilqr = IterativeLQR(acrobot, task, model, hori, reuse_feedback=5, 
            verbose=True)
    return ilqr

#@memory.cache
def run_experiment(model_name, controller_name, init_state):
    if model_name == "approx_gp":
        model = train_approx_gp(50)
    elif model_name == "true":
        from acrobot_model import AcrobotModel
        model = AcrobotModel(acrobot)
    elif model_name == "mlp":
        model = train_mlp(490)
    else:
        raise ValueError("Unknown model type")


    # Now it's time to apply the controller
    task1 = ampc.Task(acrobot)
    Q = np.diag([1.0, 1.0, 1.0, 1.0])
    R = np.diag([1.0]) * 0.01
    F = np.diag([10., 10., 10., 10.])*10.0
    task1.set_quad_cost(Q, R, F)
    #task1.set_ctrl_bound("u", -20, 20)
    from autompc.tasks.quad_cost import QuadCost
    cost = QuadCost(acrobot, Q, R)
    from autompc.tasks.task import Task as Task2
    task2 = Task2(acrobot)
    task2.set_cost(cost)

    if controller_name == "ilqr":
        con = init_ilqr(model, task1, hori=20)
    elif controller_name == "lqr":
        con = create_lqr_controller(task2)
    else:
        raise ValueError("Unknown controler type")

    # just give a random initial state
    sim_traj = ampc.zeros(acrobot, 1)
    #x = np.array([0.01, 0, 0, 0])
    x = init_state
    sim_traj[0].obs[:] = x
    us = []


    constate = con.traj_to_state(sim_traj[:1])
    #model_state = model.traj_to_state(sim_traj[:1])
    for step in range(300):
        u, constate = con.run(constate, sim_traj[-1].obs)
        #u = np.zeros(1,)
        print('u = ', u, 'state = ', sim_traj[-1].obs)
        x = dt_acrobot_dynamics(sim_traj[-1].obs, u, dt)
        #model_state = model.pred(model_state, u)
        #x = model_state[:cartpole.obs_dim]
        # x = model.pred(sim_traj[-1].obs, u)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        us.append(u)
    return sim_traj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["approx_gp", "mlp", "true"], 
            default = "approx_gp", help="Specify which system id model to use")
    parser.add_argument("--controller", type=str, choices=["ilqr", "lqr"], 
            default = "ilqr", help="Specify which nonlinear controller to use")
    parser.add_argument("--init_angle", type=float, default=0.1,
            help="Specify the initial angle for the simulation.")
    args = parser.parse_args()

    dt = 0.05
    acrobot.dt = dt

    init_state = np.array([args.init_angle, 0.0, 0.0, 0.0])
    sim_traj = run_experiment(args.model, args.controller, init_state)
    set_trace()

    print(sim_traj.obs)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ani = animate_acrobot(fig, ax, dt, sim_traj)
    #ani.save("out/cartpole_test/aug31_02.mp4")
    plt.show()

if __name__ == "__main__":
    main()
