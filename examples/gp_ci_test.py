# Created by William Edwards

from pdb import set_trace
import numpy as np
import autompc as ampc
from autompc.tasks.quad_cost import QuadCost
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from joblib import Memory
from func_timeout import func_timeout, FunctionTimedOut
memory = Memory("cache")
import time

rng = np.random.RandomState(42)

cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
dt = 0.01

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
    return np.array([omega,
            1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                - (m_c+m_p+m_p)*g*np.sin(theta)
                - b*omega),
            dx,
            1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                (L*omega**2 + g*np.cos(theta)))])

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
    #y[0] += np.pi
    #sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    #if not sol.success:
    #    raise Exception("Integration failed due to {}".format(sol.message))
    #y = sol.y.reshape((4,))
    y += dt * cartpole_simp_dynamics(y,u[0],g,m,L,b)
    #y[0] -= np.pi
    return y


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



dt = 0.01
cartpole.dt = dt

umin = -20.0
umax = 20.0
udmax = 0.25

# Generate trajectories for training
num_trajs = 500

@memory.cache
def gen_trajs(traj_len, num_trajs=num_trajs, dt=dt, rand_contr_prob=1.0, seed=42):
    rng = np.random.default_rng(seed)
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

X = np.concatenate([traj.obs for traj in trajs2])
set_trace()

from cartpole_model import CartpoleModel
true_dyn = CartpoleModel(cartpole)

Q = np.eye(4)
R = 0.01 * np.eye(1)

cost = QuadCost(cartpole, Q, R)

from autompc.tasks.task import Task

task = Task(cartpole)
task.set_cost(cost)
task.set_ctrl_bound("u", -20.0, 20.0)

from autompc.tasks.quad_cost_transformer import QuadCostTransformer
from autompc.pipelines import FixedControlPipeline
from autompc.sysid import Koopman
from autompc.control import FiniteHorizonLQR


pipeline = FixedControlPipeline(cartpole, task, Koopman, FiniteHorizonLQR, 
    [QuadCostTransformer])

from autompc.control_evaluation import CrossDataEvaluator, FixedModelEvaluator
from autompc.control_evaluation import FixedInitialMetric

init_states = [np.array([0.5, 0.0, 0.0, 0.0])]

metric = FixedInitialMetric(cartpole, task, init_states, sim_iters=100,
        sim_time_limit=10.0)

training_trajs = trajs[:100]
validation_trajs = trajs[100:]

from autompc.evaluators import HoldoutEvaluator
from autompc.metrics import RmseKstepMetric

@memory.cache
def get_cross_data_evaluator(*args, **kwargs):
    return CrossDataEvaluator(*args, **kwargs)


from autompc.sysid import (GaussianProcess, 
        LargeGaussianProcess, 
        ApproximateGaussianProcess, MLP)
from autompc.control import IterativeLQR

@memory.cache
def sample_approx_gp_inner(num_trajs, seed):
    cs = ApproximateGaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, ApproximateGaussianProcess, cfg)
    traj_sample = gen_trajs(traj_len=200, num_trajs=num_trajs, seed=seed,
            rand_contr_prob=0.5)
    model.train(traj_sample)
    return model.get_parameters()

def sample_approx_gp(num_trajs, seed):
    cs = ApproximateGaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, ApproximateGaussianProcess, cfg)
    params = sample_approx_gp_inner(num_trajs, seed)
    model.set_parameters(params)
    return model

@memory.cache
def sample_mlp_inner(num_trajs, seed, rc_prob):
    cs = MLP.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    traj_sample = gen_trajs(traj_len=200, num_trajs=num_trajs, 
            seed=seed, rand_contr_prob=rc_prob)
    model = ampc.make_model(cartpole, MLP, cfg)
    model.train(trajs2[-num_trajs:])
    return model.get_parameters()

def sample_mlp(num_trajs, seed, rc_prob=1.0):
    cs = MLP.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, MLP, cfg)
    params = sample_mlp_inner(num_trajs, seed, rc_prob)
    model.set_parameters(params)
    return model

#from scipy import stats
#def get_ucb(sample, p, alpha):
#    sample.sort()
#    s = int(stats.binom.ppf(1 - alpha, len(sample), p))
#    if s < len(sample):
#        upper = sample[s]
#    else:
#        upper = float("inf")
#    return upper



evaluator_true = FixedModelEvaluator(cartpole, task, metric, training_trajs, 
        sim_model=true_dyn)
cs = pipeline.get_configuration_space()
cfg1 = cs.get_default_configuration()
cfg1["_controller:horizon"] = 1000
eval_true = evaluator_true(pipeline)
true_cost = eval_true(cfg1)
print(f"True dynamics cost is {true_cost}")

gp = sample_approx_gp(num_trajs=5, seed=77)

n_trials = 100
gp_costs = []
for i in range(n_trials):
    gp.pred = gp.get_sampler()
    evaluator_gp = FixedModelEvaluator(cartpole, task, metric, training_trajs, 
            sim_model=gp)
    eval_gp = evaluator_gp(pipeline)
    gp_costs.append(eval_gp(cfg1))

sorted_costs = gp_costs[:]
sorted_costs.sort()
p90 = sorted_costs[int(len(sorted_costs)*0.9)]
p75 = sorted_costs[int(len(sorted_costs)*0.75)]
p50 = sorted_costs[int(len(sorted_costs)*0.5)]

fig = plt.figure()
ax = fig.gca()

ax.set_title(f"Cost Distribution")
ax.set_xlabel("Cost")
ax.set_ylabel("Count")
ax.axvline(x=true_cost, color="r", label="True dynamics cost")
ax.axvline(x=p90, color="g", label="90th percentile")
ax.axvline(x=p75, color="y", label="75th percentile")
ax.axvline(x=p50, color="k", label="50th percentile")
ax.hist(gp_costs, bins=np.arange(0, 2000, 25))
ax.legend()

plt.show()

set_trace()
