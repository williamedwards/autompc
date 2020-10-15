"""Just test if my MLP code is functioning as expected
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory
from pdb import set_trace

import autompc as ampc
from test_shared import *


memory = Memory("cache")
linsys = ampc.System(['x', 'v'], ['a'])
linsys.dt = 0.1
pendulum = ampc.System(["ang", "angvel"], ["torque"])
pendulum.dt = 0.05
cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
cartpole.dt = 0.01
planar_drone = ampc.System(["x", "dx", "y", "dy", "theta", "omega"], ["u1", "u2"])
planar_drone = 0.05

dt = 0.01

def cartpole_simp_dynamics(y, u, g = 9.8, m = 1, L = 1, b = 0.1):
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
    return np.array([omega,
            g * np.sin(theta)/L - b * omega / (m*L**2) + u * np.cos(theta)/L,
            dx,
            u])

def dt_cartpole_dynamics(y,u,dt,g=9.8,m=1,L=1,b=1.0):
    y = np.copy(y)

    y += dt * cartpole_simp_dynamics(y,u[0],g,m,L,b)
    return y

cartpole.dt = dt

from cartpole_model import CartpoleModel
from autompc.control import FiniteHorizonLQR
from autompc.sysid.dummy_linear import DummyLinear

def get_generation_controller():
    truedyn = CartpoleModel(cartpole)
    _, A, B = truedyn.pred_diff(np.zeros(4,), np.zeros(1))
    model = DummyLinear(cartpole, A, B)
    Q = np.eye(4)
    R = 0.01 * np.eye(1)

    from autompc.tasks.quad_cost import QuadCost
    cost = QuadCost(cartpole, Q, R)

    from autompc.tasks.task import Task

    task = Task(cartpole)
    task.set_cost(cost)
    task.set_ctrl_bound("u", -20.0, 20.0)
    cs = FiniteHorizonLQR.get_configuration_space(cartpole, task, model)
    cfg = cs.get_default_configuration()
    cfg["horizon"] = 1000
    con = ampc.make_controller(cartpole, task, model, FiniteHorizonLQR, cfg)
    return con

umin = -20.0
umax = 20.0
udmax = 0.25

# Generate trajectories for training
num_trajs = 500

@memory.cache
def gen_trajs(traj_len, num_trajs=num_trajs, dt=dt, rand_contr_prob=1.0):
    rng = np.random.default_rng(49)
    trajs = []
    con = get_generation_controller()
    for _ in range(num_trajs):
        theta0 = rng.uniform(-1.0, 1.0, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(cartpole, traj_len)
        traj.obs[:] = y
        if rng.random() < rand_contr_prob:
            actuate = False
        else:
            actuate = True
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
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs
trajs = gen_trajs(4)
trajs2 = gen_trajs(200)
trajs4 = gen_trajs(200, rand_contr_prob = 0.0)



from autompc.sysid import MLP
@memory.cache
def train_mlp_inner(num_trajs):
    cs = MLP.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, MLP, cfg)
    model.train(trajs2[-num_trajs:])
    return model.get_parameters()

def train_mlp(num_trajs):
    cs = MLP.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, MLP, cfg)
    params = train_mlp_inner(num_trajs)
    model.set_parameters(params)
    #model.net = model.net.to("cpu")
    #model._device = "cpu"
    return model



def test_cartpole():
    """Test the MLP model and potentially cem on cartpole problem"""
    from autompc.sysid import MLP
    # collect trajectories
    dt = 0.05
    umin, umax = -2, 2
    trajs = collect_cartpole_trajs(dt, umin, umax, num_trajs=10)
    n_hidden, hidden_size, nonlintype, n_iter, n_batch, lr = 2, 32, 'relu', 10, 64, 1e-3
    model = MLP(cartpole, n_hidden, hidden_size, nonlintype, n_iter, n_batch, lr)
    model.train(trajs)
    # make predictions...
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([np.pi, 0, 0, 0])
    sim_traj[0].obs[:] = x
    us = []

    for step in range(200):
        u = np.random.random(1)
        x = dt_cartpole_dynamics(sim_traj[-1].obs, u, dt)
        newx = model.pred(sim_traj[-1].obs, u)
        newx, jx, ju = model.pred_diff(sim_traj[-1].obs, u)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        us.append(u)


def test_mlp_scatter():
    model = train_mlp(80)
    fig, axs = plt.subplots(1, 1)
    #axs = axs.flatten()
    for i, state in enumerate(["theta", "omega", "x", "dx"]):
        true = []
        pred = []
        for traj in trajs2[:10]:
            for j in range(1, len(traj)):
                true.append(traj[j, state])
                modelstate = model.traj_to_state(traj[:j])
                p = model.pred(modelstate, traj[j].ctrl[:])
                pred.append(p[i])
        if i == 1:
            #ax = axs[i]
            ax = axs
            ax.scatter(true, pred, s=0.1)
            ax.set_xlabel("true {}".format(state))
            ax.set_ylabel("predicted {}".format(state))
            xlow, xhigh = ax.get_xlim()
            points = np.linspace(xlow, xhigh, 50)
            ax.plot(points, points, "k--")
            ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

def test_mlp_scatter_lqr():
    model = train_mlp(80)
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    for i, state in enumerate(["theta", "omega", "x", "dx"]):
        true = []
        pred = []
        for traj in trajs4[:50]:
            for j in range(1, len(traj)):
                true.append(traj[j, state])
                modelstate = model.traj_to_state(traj[:j])
                p = model.pred(modelstate, traj[j].ctrl[:])
                pred.append(p[i])
        ax = axs[i]
        ax.scatter(true, pred, s=0.1)
        ax.set_xlabel("true {}".format(state))
        ax.set_ylabel("predicted {}".format(state))
        xlow, xhigh = ax.get_xlim()
        points = np.linspace(xlow, xhigh, 50)
        ax.plot(points, points, "k--")
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

def test_mlp_rollout():
    model = train_mlp(450)

    # Generate actuated trajectory
    con = get_generation_controller()
    traj = ampc.zeros(cartpole, 200)
    traj[0, "theta"] = 0.5
    y = traj[0].obs[:]
    constate = con.traj_to_state(traj[:1])
    for i in range(len(traj)):
        traj[i].obs[:] = y
        u, constate = con.run(constate, y)
        y = dt_cartpole_dynamics(y, u, dt)
        traj[i].ctrl[:] = u

    # Compute model rollout
    pred_traj = ampc.zeros(cartpole, len(traj))
    pred_traj[0].obs[:] = traj[0].obs
    modelstate = model.traj_to_state(pred_traj[:1])
    for i in range(len(traj)-1):
        u = traj[i].ctrl[:]
        pred_traj[i].ctrl[:] = u
        modelstate = model.pred(modelstate, u)
        pred_traj[i+1].obs[:] = modelstate

    # Plot model rollout
    fig, axs = plt.subplots(2,2)
    axs = axs.flatten()
    for i, state in enumerate(["theta", "omega", "x", "dx"]):
        ax = axs[i]
        times = np.array(list(range(len(traj)))) * 0.01
        ax.plot(times, traj.obs[:, i], "g")
        ax.plot(times, pred_traj.obs[:, i], "r")
        ax.legend(["True traj", "MLP rollout"])
        ax.set_xlabel("Time")
        ax.set_ylabel(state)
    plt.tight_layout()
    plt.show()

def test_mlp_rollout_closed_loop():
    model = train_mlp(450)

    # Generate actuated trajectory
    con = get_generation_controller()
    traj = ampc.zeros(cartpole, 200)
    traj[0, "theta"] = 0.5
    y = traj[0].obs[:]
    constate = con.traj_to_state(traj[:1])
    for i in range(len(traj)):
        traj[i].obs[:] = y
        u, constate = con.run(constate, y)
        y = dt_cartpole_dynamics(y, u, dt)
        traj[i].ctrl[:] = u

    # Compute model rollout
    pred_traj = ampc.zeros(cartpole, len(traj))
    pred_traj[0].obs[:] = traj[0].obs
    modelstate = model.traj_to_state(pred_traj[:1])
    constate = con.traj_to_state(pred_traj[:1])
    for i in range(len(traj)-1):
        u, constate = con.run(constate, pred_traj[i].obs[:])
        pred_traj[i].ctrl[:] = u
        modelstate = model.pred(modelstate, u)
        pred_traj[i+1].obs[:] = modelstate

    # Plot model rollout
    fig, axs = plt.subplots(2,2)
    axs = axs.flatten()
    for i, state in enumerate(["theta", "omega", "x", "dx"]):
        ax = axs[i]
        times = np.array(list(range(len(traj)))) * 0.01
        ax.plot(times, traj.obs[:, i], "g")
        ax.plot(times, pred_traj.obs[:, i], "r")
        ax.legend(["True traj", "MLP rollout"])
        ax.set_xlabel("Time")
        ax.set_ylabel(state)
    plt.tight_layout()
    plt.show()

def test_mlp_accuracy():
    from autompc.evaluators import HoldoutEvaluator, FixedSetEvaluator
    from autompc.metrics import RmseKstepMetric
    from autompc.graphs import KstepGrapher, InteractiveEvalGrapher
    from autompc.sysid import MLP

    metric = RmseKstepMetric(cartpole, k=10)
    grapher = InteractiveEvalGrapher(cartpole, logscale=True)
    grapher2 = KstepGrapher(cartpole, kmax=50, kstep=5, evalstep=10)

    rng = np.random.default_rng(42)

    dt = 0.05
    umin, umax = -2, 2
    trajs2 = collect_cartpole_trajs(dt, umin, umax, num_trajs=500)

    evaluator = FixedSetEvaluator(cartpole, trajs2[:10], metric, rng, 
            training_trajs=trajs2[10:]) 
    evaluator.add_grapher(grapher)
    evaluator.add_grapher(grapher2)
    cs = MLP.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    eval_score, _, graphs = evaluator(MLP, cfg)
    print("eval_score = {}".format(eval_score))
    fig = plt.figure()
    graph = graphs[0]
    graph.set_obs_lower_bound("theta", -0.2)
    graph.set_obs_upper_bound("theta", 0.2)
    graph.set_obs_lower_bound("omega", -0.2)
    graph.set_obs_upper_bound("omega", 0.2)
    graph.set_obs_lower_bound("dx", -0.2)
    graph.set_obs_upper_bound("dx", 0.2)
    graph.set_obs_lower_bound("x", -0.2)
    graph.set_obs_upper_bound("x", 0.2)
    graphs[0](fig)
    fig2 = plt.figure()
    graph = graphs[1]
    # graph.set_obs_lower_bound("theta", -0.2)
    # graph.set_obs_upper_bound("theta", 0.2)
    # graph.set_obs_lower_bound("omega", -0.2)
    # graph.set_obs_upper_bound("omega", 0.2)
    # graph.set_obs_lower_bound("dx", -0.2)
    # graph.set_obs_upper_bound("dx", 0.2)
    # graph.set_obs_lower_bound("x", -0.2)
    # graph.set_obs_upper_bound("x", 0.2)
    graphs[1](fig2)
    plt.show()

    sys.exit(0)

def time_mlp():
    import timeit
    model = train_mlp(450)
    model.net = model.net.to("cpu")
    model._device = "cpu"

    m = 100
    t1 = timeit.Timer(lambda: model.pred_parallel(np.zeros((m,4)), np.ones((m,1))))
    t2 = timeit.Timer(lambda: model.pred_diff_parallel(np.zeros((m,4)), np.ones((m,1))))
    n = 100
    print(f"{t1.timeit(number=n)/n*1000=} ms")
    rep = t1.repeat(number=n)
    print("rep1=", [str(r/n*1000) + " ms" for r in rep])
    print(f"{t2.timeit(number=n)/n*1000=} ms")
    rep2 = t2.repeat(number=n)
    print("rep2=", [str(r/n*1000) + " ms" for r in rep2]) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, choices=["accuracy", "scatter", 
        "scatter-lqr", "rollout", "rollout-cl", "time"], 
            default='accuracy', help='Specify which test to run')
    args = parser.parse_args()
    if args.test == "accuracy":
        test_mlp_accuracy()
    if args.test == "scatter":
        test_mlp_scatter()
    if args.test == "scatter-lqr":
        test_mlp_scatter_lqr()
    if args.test == "rollout":
        test_mlp_rollout()
    if args.test == "rollout-cl":
        test_mlp_rollout_closed_loop()
    if args.test == "time":
        time_mlp()
