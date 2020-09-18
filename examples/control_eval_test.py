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


def dt_cartpole_dynamics(y,u,dt,g=9.8,m=1,L=1,b=1.0):
    #y = np.copy(y)
    y[0] += np.pi
    sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y = sol.y.reshape((4,))
    y[0] -= np.pi
    return y

dt = 0.01
cartpole.dt = dt

umin = -2.0
umax = 2.0
udmax = 0.25

# Generate trajectories for training
num_trajs = 500

@memory.cache
def gen_trajs(traj_len, num_trajs=num_trajs, dt=dt):
    rng = np.random.default_rng(49)
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

from cartpole_model import CartpoleModel
true_dyn = CartpoleModel(cartpole)

Q = np.eye(4)
R = 0.01 * np.eye(1)

cost = QuadCost(cartpole, Q, R)

from autompc.tasks.task import Task

task = Task(cartpole)
task.set_cost(cost)

from autompc.tasks.quad_cost_transformer import QuadCostTransformer
from autompc.pipelines import FixedControlPipeline
from autompc.sysid import Koopman
from autompc.control import FiniteHorizonLQR


pipeline = FixedControlPipeline(cartpole, task, Koopman, FiniteHorizonLQR, 
    [QuadCostTransformer])

from autompc.control_evaluation import CrossDataEvaluator, FixedModelEvaluator
from autompc.control_evaluation import FixedInitialMetric

init_states = [np.array([0.0, 0.0, 0.0, 0.0]),
               np.array([0.2, 0.0, 0.0, 0.0]),
               np.array([0.4, 0.0, 0.0, 0.0]),
               np.array([0.6, 0.0, 0.0, 0.0]),
               np.array([0.8, 0.0, 0.0, 0.0]),
               np.array([1.0, 0.0, 0.0, 0.0]),
               np.array([1.2, 0.0, 0.0, 0.0]),
               np.array([1.4, 0.0, 0.0, 0.0]),
               np.array([1.6, 0.0, 0.0, 0.0]),
               np.array([1.8, 0.0, 0.0, 0.0]),
               np.array([2.0, 0.0, 0.0, 0.0])]

metric1 = FixedInitialMetric(cartpole, task, init_states[:4], sim_iters=1000)
metric2 = FixedInitialMetric(cartpole, task, init_states, sim_iters=1000)

training_trajs = trajs[:100]
validation_trajs = trajs[100:]

from autompc.evaluators import HoldoutEvaluator
from autompc.metrics import RmseKstepMetric

@memory.cache
def get_cross_data_evaluator(*args, **kwargs):
    return CrossDataEvaluator(*args, **kwargs)

model_metric = RmseKstepMetric(cartpole)
evaluator1a = FixedModelEvaluator(cartpole, task, metric1, training_trajs, 
        sim_model=true_dyn)
evaluator2a = get_cross_data_evaluator(cartpole, task, metric1, 
        HoldoutEvaluator, {"holdout_prop" : 0.25, "primary_metric" : model_metric},
        rng, training_trajs, validation_trajs, tuning_iters=1)
evaluator1b = FixedModelEvaluator(cartpole, task, metric2, training_trajs, 
        sim_model=true_dyn)
evaluator2b = get_cross_data_evaluator(cartpole, task, metric2, 
        HoldoutEvaluator, {"holdout_prop" : 0.25, "primary_metric" : model_metric},
        rng, training_trajs, validation_trajs, tuning_iters=1)


cs = pipeline.get_configuration_space()
print(cs)
cfg1 = cs.get_default_configuration()
cfg2 = cs.get_default_configuration()
cfg3 = cs.get_default_configuration()
cfg4 = cs.get_default_configuration()
cfg5 = cs.get_default_configuration()
cfg6 = cs.get_default_configuration()
cfg7 = cs.get_default_configuration()
cfg8 = cs.get_default_configuration()
cfg9 = cs.get_default_configuration()
cfg10 = cs.get_default_configuration()
cfg11 = cs.get_default_configuration()
cfg12 = cs.get_default_configuration()
cfg13 = cs.get_default_configuration()
cfg14 = cs.get_default_configuration()
cfg15 = cs.get_default_configuration()
cfg16 = cs.get_default_configuration()
cfg1["_controller:horizon"] = 1000
cfg2["_controller:horizon"] = 100
cfg3["_task_transformer_0:theta_log10Qgain"] = -2.0
cfg3["_controller:horizon"] = 1000
cfg4["_task_transformer_0:omega_log10Qgain"] = 2.0
cfg4["_controller:horizon"] = 1000
cfg5["_task_transformer_0:omega_log10Qgain"] = 2.0
cfg5["_controller:horizon"] = 100
cfg6["_task_transformer_0:u_log10Rgain"] = -2.0
cfg6["_controller:horizon"] = 1000
cfg7["_task_transformer_0:u_log10Rgain"] = -2.0
cfg7["_controller:horizon"] = 100
cfg8["_task_transformer_0:u_log10Rgain"] = -2.0
cfg8["_task_transformer_0:omega_log10Qgain"] = 2.0
cfg8["_controller:horizon"] = 1000
cfg9["_task_transformer_0:u_log10Rgain"] = -2.0
cfg9["_task_transformer_0:omega_log10Qgain"] = 2.0
cfg9["_controller:horizon"] = 100
cfg10["_task_transformer_0:u_log10Rgain"] = -2.0
cfg10["_task_transformer_0:dx_log10Qgain"] = 2.0
cfg10["_controller:horizon"] = 100
cfg11["_task_transformer_0:u_log10Rgain"] = -3.0
cfg11["_task_transformer_0:omega_log10Qgain"] = 4.0
cfg11["_controller:horizon"] = 1000
cfg12["_task_transformer_0:u_log10Rgain"] = -3.0
cfg12["_task_transformer_0:omega_log10Qgain"] = 4.0
cfg12["_controller:horizon"] = 100
cfg13["_task_transformer_0:u_log10Rgain"] = -3.0
cfg13["_task_transformer_0:omega_log10Qgain"] = 0.5
cfg13["_controller:horizon"] = 200
cfg14["_task_transformer_0:u_log10Rgain"] = -0.5
cfg14["_task_transformer_0:omega_log10Qgain"] = 3.0
cfg14["_controller:horizon"] = 800
cfg15["_task_transformer_0:u_log10Rgain"] = -2.0
cfg15["_task_transformer_0:dx_log10Qgain"] = 4.0
cfg15["_controller:horizon"] = 1000
cfg16["_task_transformer_0:dx_log10Qgain"] = 2.0
cfg16["_task_transformer_0:omega_log10Qgain"] = 2.0
cfg16["_controller:horizon"] = 100

cfgs = [cfg1, cfg2, cfg3, cfg4, cfg5, cfg6, cfg7, cfg8, cfg9, cfg10]
eval_cfg1a = evaluator1a(pipeline)
eval_cfg2a = evaluator2a(pipeline)
eval_cfg1b = evaluator1b(pipeline)
eval_cfg2b, sim_model = evaluator2b(pipeline, ret_model=True)
#evals1a = [eval_cfg1a(cfg) for cfg in cfgs]
#evals2a = [eval_cfg2a(cfg) for cfg in cfgs]
#evals1b = [eval_cfg1b(cfg) for cfg in cfgs]
#evals2b = [eval_cfg2b(cfg) for cfg in cfgs]
#print(f"{evals1a=}")
#print(f"{evals2a=}")
#print(f"{evals1b=}")
#print(f"{evals2b=}")
#
#fig = plt.figure()
#ax = fig.gca()
#ax.scatter(evals1a, evals2a)
#xs = np.logspace(3.0, 6.5, 100)
#ax.plot(xs, xs, "k--")
#ax.set_xscale("log")
#ax.set_yscale("log")
#ax.set_xlabel("True Dynamics Evaluation")
#ax.set_ylabel("Cross-Data Evaluation")
#plt.show()
#
#fig = plt.figure()
#ax = fig.gca()
#ax.scatter(evals1b, evals2b)
#xs = np.logspace(3.0, 6.5, 100)
#ax.plot(xs, xs, "k--")
#ax.set_xscale("log")
#ax.set_yscale("log")
#ax.set_xlabel("True Dynamics Evaluation")
#ax.set_ylabel("Cross-Data Evaluation")
#plt.show()

controller, model = pipeline(cfg1, training_trajs)
_, costs1 = metric2(controller, true_dyn, ret_detailed=True)
_, costs2 = metric2(controller, sim_model, ret_detailed=True)
angles = [state[0] for state in init_states]

fig = plt.figure()
ax = fig.gca()
ax.plot(angles[1:], costs1[1:], label="True dynamics")
ax.plot(angles[1:], costs2[1:], label="Cross-data")
ax.set_xlabel("Starting angle")
ax.set_ylabel("Cost")
ax.set_yscale("log")
ax.legend()
plt.show()

