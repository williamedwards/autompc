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
acrobot2 = ampc.System(["sin1", "cos1", "sin2", "cos2", "dtheta1", "dtheta2"], ["u"])
#system = acrobot2

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
    #if y[0] < -2 * np.pi:
    #    y[0] += 2 * np.pi
    #if y[0] > 2 * np.pi:
    #    y[0] -= 2 * np.pi
    #if y[1] < -2 * np.pi:
    #    y[1] += 2 * np.pi
    #if y[1] > 2 * np.pi:
    #    y[1] -= 2 * np.pi
    return y

def dt_acrobot2_dynamics(x, u, dt):
    y = np.array([np.arctan2(x[0], x[1]), np.arctan2(x[2], x[3]), x[4], x[5]])
    y[0] += np.pi
    y += dt * acrobot_dynamics(y, u[0])
    y[0] -= np.pi
    newx = np.array([np.sin(y[0]), np.cos(y[0]), np.sin(y[1]), np.cos(y[1]), y[2], y[3]])
    return newx

def traj_acrobot_to_acrobot2(traj):
    traj2 = ampc.zeros(acrobot2, len(traj))
    traj2.obs[:, 0] = np.sin(traj.obs[:,0])
    traj2.obs[:, 1] = np.cos(traj.obs[:,0])
    traj2.obs[:, 2] = np.sin(traj.obs[:,1])
    traj2.obs[:, 3] = np.cos(traj.obs[:,1])
    traj2.obs[:, 4] = traj.obs[:,2]
    traj2.obs[:, 5] = traj.obs[:,3]
    traj2.ctrls[:] = traj.ctrls[:]
    return traj2
    
def traj_acrobot2_to_acrobot(traj2):
    traj = ampc.zeros(acrobot, len(traj2))
    traj.obs[:, 0] = np.arctan2(traj2.obs[:, 0], traj2.obs[:, 1])
    traj.obs[:, 1] = np.arctan2(traj2.obs[:, 2], traj2.obs[:, 3])
    traj.obs[:, 2] = traj2.obs[:, 4]
    traj.obs[:, 3] = traj2.obs[:, 5]
    traj.ctrls[:, :] = traj2.ctrls
    return traj

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
acrobot2.dt = dt

umin = -20.0
umax = 20.0
udmax = 0.25

# Generate trajectories for training
num_trajs = 500

#from cartpole_model import CartpoleModel
#from autompc.control import FiniteHorizonLQR
#from autompc.sysid.dummy_linear import DummyLinear
#

from autompc.sysid import (GaussianProcess, 
        LargeGaussianProcess, 
        ApproximateGaussianProcess, MLP)
from autompc.control import IterativeLQR, FiniteHorizonLQR

def init_ilqr(system, model, task, hori=40, reuse_feedback=1):
    mode = 'auglag'
    ilqr = IterativeLQR(system, task, model, hori, reuse_feedback=reuse_feedback, 
            verbose=True)
    return ilqr

def get_generation_controller():
    from acrobot_model import AcrobotModel
    model = AcrobotModel(acrobot)

    Q = 0.01 * np.eye(4)
    R = 0.01 * np.eye(1)
    F = 20.0 * np.eye(4)
    Q2 = np.eye(4)
    R2 = np.eye(1)
    F2 = np.eye(4)
    from autompc.tasks.quad_cost import QuadCost
    cost = QuadCost(acrobot, Q, R, F)
    cost2 = QuadCost(acrobot, Q2, R2, F2)
    from autompc.tasks.task import Task
    task1 = Task(acrobot)
    task1.set_cost(cost)
    task1.set_ctrl_bound("u", -100, 100)
    con = init_ilqr(model, task1, hori=20, reuse_feedback=5)
    return con

# Generate trajectories for training
num_trajs = 500

@memory.cache
def gen_trajs(traj_len, num_trajs=num_trajs, dt=dt, rand_contr_prob=1.0):
    rng = np.random.default_rng(49)
    trajs = []
    con = get_generation_controller()
    for _ in range(num_trajs):
        if rng.random() < rand_contr_prob:
            actuate = False
            theta0 = rng.uniform(-0.01, 0.01, 1)[0]
        else:
            actuate = True
            theta0 = rng.uniform(-3.0, 3.0, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(acrobot, traj_len)
        traj.obs[:] = y
        if actuate:
            constate = con.traj_to_state(traj[:1])
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
        if np.isnan(traj.obs).any() or np.isnan(traj.ctrls).any():
            continue
        trajs.append(traj)
    return trajs
trajs = gen_trajs(4)
trajs2 = gen_trajs(40)
#trajs3 = gen_trajs(200, rand_contr_prob = 0.5)
#trajs4 = gen_trajs(200, rand_contr_prob = 0.0)


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
def train_mlp_inner(system, num_trajs, change_state, seed, n_train_iters=50):
    cs = MLP.get_configuration_space(system)
    cfg = cs.get_default_configuration()
    cfg["nonlintype"] = "relu"
    cfg["n_hidden_layers"] = "2"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    model = ampc.make_model(system, MLP, cfg, 
            n_train_iters=n_train_iters)
    torch.manual_seed(seed)
    if change_state:
        train_trajs = [traj_acrobot_to_acrobot2(traj) for traj in trajs2]
    else:
        train_trajs = trajs2
    model.train(train_trajs[-num_trajs:])
    return model.get_parameters()

def train_mlp(system, num_trajs, change_state, seed=42):
    cs = MLP.get_configuration_space(system)
    cfg = cs.get_default_configuration()
    cfg["nonlintype"] = "relu"
    cfg["n_hidden_layers"] = "2"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    model = ampc.make_model(system, MLP, cfg)
    params = train_mlp_inner(system, num_trajs, change_state,
            seed, n_train_iters=20)
    model.set_parameters(params)
    return model




def fd_jac(func, x, dt=1e-4):
    res = func(x)
    jac = np.empty((res.size, x.size))
    for i in range(x.size):
        xp = np.copy(x)
        xp[i] += dt
        resp = func(xp)
        jac[:,i] = (resp - res) / dt
    return jac

#@memory.cache
def run_experiment(model_name, controller_name, init_state, change_state):
    if change_state:
        system = acrobot2
    else:
        system = acrobot
    if model_name == "true":
        from acrobot_model import AcrobotModel
        model = AcrobotModel(acrobot)
    elif model_name == "mlp":
        model = train_mlp(system, 600, change_state)
    else:
        raise ValueError("Unknown model type")


    ## Test model gradients
    #state = np.zeros(4,)
    #state[1] = 1.0
    #ctrl = np.ones(1,)
    #pred, state_jac, ctrl_jac = model.pred_diff(state, ctrl)
    #state_jac2 = fd_jac(lambda y: model.pred(y, ctrl), state)
    #ctrl_jac2 = fd_jac(lambda y: model.pred(state, y), ctrl)
    #print(f"{state_jac=}")
    #print(f"{state_jac2=}")
    #print(f"{ctrl_jac=}")
    #print(f"{ctrl_jac2=}")


    # Now it's time to apply the controller
    #task1 = ampc.Task(acrobot)
    #Q = np.diag([10.0, 10.0, 0.0001, 0.0001])
    #R = np.diag([1.0]) * 0.0001
    #F = np.diag([100., 100., 10000., 10000.])*10000.0
    #Q = np.eye(4)
    #R = 0.01 * np.eye(1)
    #F = 20.0 * np.eye(4)
    #Q2 = np.eye(4)
    #R2 = np.eye(1)
    #F2 = np.eye(4)
    from autompc.tasks.quad_cost import QuadCost
    #cost = QuadCost(acrobot, Q, R, F)
    #cost2 = QuadCost(acrobot, Q2, R2, F2)
    from autompc.tasks.task import Task
    #task1 = Task(acrobot)
    #task1.set_cost(cost)
    #task1.set_ctrl_bound("u", -100, 100)

    if change_state:
        Q = 0.00 * np.eye(6)
        R = 0.01 * np.eye(1)
        F = 100.0 * np.eye(6)
        x0 = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    else:
        Q = 0.00 * np.eye(4)
        R = 0.01 * np.eye(1)
        F = 100.0 * np.eye(4)
        x0 = np.zeros(4)
    #F[-1,-1] /= 10
    #F[-2,-2] /= 10
    cost = QuadCost(system, Q, R, F, x0=x0)
    task1 = Task(system)
    task1.set_cost(cost)
    task1.set_ctrl_bound("u", -100, 100)


    if controller_name == "ilqr":
        con = init_ilqr(system, model, task1, hori=20, reuse_feedback=-1)
    elif controller_name == "lqr":
        con = create_lqr_controller(task1)
    else:
        raise ValueError("Unknown controler type")

    # just give a random initial state
    sim_traj = ampc.zeros(system, 1)
    #x = np.array([0.01, 0, 0, 0])
    if change_state:
        x = [np.sin(init_state[0]), np.cos(init_state[0]),
                np.sin(init_state[1]), np.cos(init_state[1]),
                init_state[2], init_state[3]]
    else:
        x = init_state
    sim_traj[0].obs[:] = x
    us = []


    constate = con.traj_to_state(sim_traj[:1])
    #model_state = model.traj_to_state(sim_traj[:1])
    for step in range(300):
        u, constate = con.run(constate, sim_traj[-1].obs)
        #u = np.zeros(1,)
        print('u = ', u, 'state = ', sim_traj[-1].obs)
        if change_state:
            x = dt_acrobot2_dynamics(sim_traj[-1].obs, u, dt)
        else:
            x = dt_acrobot_dynamics(sim_traj[-1].obs, u, dt)
        #model_state = model.pred(model_state, u)
        #x = model_state[:cartpole.obs_dim]
        # x = model.pred(sim_traj[-1].obs, u)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        us.append(u)
    #print(f"Performance metric = {cost2(sim_traj)}")
    if change_state:
        sim_traj = traj_acrobot2_to_acrobot(sim_traj)
    return sim_traj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["approx_gp", "mlp", "true"], 
            default = "approx_gp", help="Specify which system id model to use")
    parser.add_argument("--controller", type=str, choices=["ilqr", "lqr", "mppi"], 
            default = "ilqr", help="Specify which nonlinear controller to use")
    parser.add_argument("--changestate", action="store_true")
    parser.add_argument("--init_angle", type=float, default=0.1,
            help="Specify the initial angle for the simulation.")
    args = parser.parse_args()

    #dt = 0.05
    #acrobot.dt = dt

    init_state = np.array([args.init_angle, 0.0, 0.0, 0.0])
    sim_traj = run_experiment(args.model, args.controller, init_state, args.changestate)

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
