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

dt = 0.01
cartpole.dt = dt

umin = -20.0
umax = 20.0
udmax = 0.25

# Generate trajectories for training
num_trajs = 500

@memory.cache
def gen_trajs(traj_len, num_trajs=num_trajs, dt=dt, seed=49):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(num_trajs):
        theta0 = rng.uniform(-0.002, 0.002, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(cartpole, traj_len)
        for i in range(traj_len):
            traj[i].obs[:] = y
            #if u[0] > umax:
            #    u[0] = umax
            #if u[0] < umin:
            #    u[0] = umin
            #u += rng.uniform(-udmax, udmax, 1)
            u  = rng.uniform(umin, umax, 1)
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs
trajs = gen_trajs(4)
trajs2 = gen_trajs(200)

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
    traj_sample = gen_trajs(traj_len=200, num_trajs=num_trajs, seed=seed)
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
def sample_mlp_inner(num_trajs, seed):
    cs = MLP.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, MLP, cfg)
    model.train(trajs2[-num_trajs:])
    return model.get_parameters()

def sample_mlp(num_trajs, seed):
    cs = MLP.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    model = ampc.make_model(cartpole, MLP, cfg)
    params = sample_mlp_inner(num_trajs, seed)
    model.set_parameters(params)
    return model




evaluator_true = FixedModelEvaluator(cartpole, task, metric, training_trajs, 
        sim_model=true_dyn)
cs = pipeline.get_configuration_space()
cfg1 = cs.get_default_configuration()
cfg1["_controller:horizon"] = 1000
eval_true = evaluator_true(pipeline)
true_cost = eval_true(cfg1)
print(f"True dynamics cost is {true_cost}")


@memory.cache
def get_gp_costs(num_trajs, num_samples):
    rng = np.random.default_rng(42)
    gp_costs = []
    for i in range(num_samples):
        seed = rng.integers(1 << 30)
        gp = sample_approx_gp(num_trajs=num_trajs, seed=seed)
        evaluator_gp = FixedModelEvaluator(cartpole, task, metric, training_trajs, 
                sim_model=gp)
        eval_gp = evaluator_gp(pipeline)
        gp_costs.append(eval_gp(cfg1))
    return gp_costs

@memory.cache
def get_mlp_costs(num_trajs, num_samples):
    rng = np.random.default_rng(42)
    gp_costs = []
    for i in range(num_samples):
        seed = rng.integers(1 << 30)
        gp = sample_mlp(num_trajs=num_trajs, seed=seed)
        evaluator_gp = FixedModelEvaluator(cartpole, task, metric, training_trajs, 
                sim_model=gp)
        eval_gp = evaluator_gp(pipeline)
        gp_costs.append(eval_gp(cfg1))
    return gp_costs

from noisy_cartpole_model import NoisyCartpoleModel
@memory.cache
def get_noisy_dyn_costs(noise_factro, num_samples):
    rng = np.random.default_rng(42)
    costs = []
    model = NoisyCartpoleModel(cartpole, rng, noise_factor)
    for i in range(num_samples):
        evaluator = FixedModelEvaluator(cartpole, task, metric, training_trajs, 
                sim_model=model)
        eval = evaluator(pipeline)
        costs.append(eval(cfg1))
    return costs

gp_costss = []
num_trajss = [5, 10, 20, 40, 80]
#num_trajss = [5]
for num_trajs in num_trajss:
    print("===== num_trajs={} ======".format(num_trajs))
    gp_costs = get_mlp_costs(num_trajs, 100)
    gp_costss.append(gp_costs)

    print(f"GP surrogate dynamics costs are {gp_costs}")

for num_trajs, gp_costs in zip(num_trajss, gp_costss):
    fig = plt.figure()
    ax = fig.gca()

    ax.set_title(f"Cost Distribution for trainsize={num_trajs} trajs")
    ax.set_xlabel("Cost")
    ax.set_ylabel("Count")
    ax.axvline(x=true_cost, color="r", label="True dynamics cost")
    ax.hist(gp_costs, bins=np.arange(0, 2000, 10))
    ax.legend()

    plt.show()

#costss = []
#noise_factors = [0.4, 0.2, 0.1, 0.05, 0.025]
##noise_factors = [0.0]
#for noise_factor in noise_factors:
#    print("===== noise_factor={} ======".format(noise_factor))
#    costs = get_noisy_dyn_costs(noise_factor, 100)
#    costss.append(costs)
#
#    #print(f"GP surrogate dynamics costs are {gp_costs}")
#
#set_trace()
#for noise_factor, costs in zip(noise_factors, costss):
#    fig = plt.figure()
#    ax = fig.gca()
#
#    ax.set_title(f"Cost Distribution for noise_factor={noise_factor}")
#    ax.set_xlabel("Cost")
#    ax.set_ylabel("Count")
#    ax.axvline(x=true_cost, color="r", label="True dynamics cost")
#    ax.hist(costs, bins=np.arange(0, 2000, 10))
#    ax.legend()
#
#    plt.show()
