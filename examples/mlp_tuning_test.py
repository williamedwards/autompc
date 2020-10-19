from pdb import set_trace
import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from joblib import Memory

from scipy.integrate import solve_ivp

memory = Memory("cache")

cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])

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

def animate_cartpole(fig, ax, dt, traj):
    ax.grid()

    line, = ax.plot([0.0, 0.0], [0.0, -1.0], 'o-', lw=2)
    time_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
    ctrl_text = ax.text(0.7, 0.85, '', transform=ax.transAxes)

    def init():
        line.set_data([0.0, 0.0], [0.0, -1.0])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        #i = min(i, ts.shape[0])
        line.set_data([traj[i,"x"], traj[i,"x"]+np.sin(traj[i,"theta"]+np.pi)], 
                [0, -np.cos(traj[i,"theta"] + np.pi)])
        time_text.set_text('t={:.2f}'.format(dt*i))
        ctrl_text.set_text("u={:.2f}".format(traj[i,"u"]))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani

dt = 0.05
cartpole.dt = dt

umin = -20.0
umax = 20.0
udmax = 0.25

# Generate trajectories for training
num_trajs = 500

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

# Generate trajectories for training
num_trajs = 500

@memory.cache
def gen_trajs(traj_len, num_trajs=num_trajs, dt=dt, rand_contr_prob=1.0,
        init_min = [-1.0, 0.0, 0.0, 0.0], init_max=[1.0, 0.0, 0.0, 0.0]):
    rng = np.random.default_rng(49)
    trajs = []
    con = get_generation_controller()
    for _ in range(num_trajs):
        state0 = [rng.uniform(minval, maxval, 1)[0] for minval, maxval 
                in zip(init_min, init_max)]
        y = state0[:]
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

def main():
    trajs = gen_trajs(4)
    trajs2 = gen_trajs(200)
    trajs3 = gen_trajs(200, rand_contr_prob = 0.5)

    from autompc.sysid import ARX, Koopman, MLP

    Model = MLP

    cs = Model.get_configuration_space(cartpole)
    s = cs.get_default_configuration()
    s["n_train_iters"] = 11
    #model = ampc.make_model(cartpole, Model, s)
    #model.train(trajs3)

    from autompc.evaluators import HoldoutEvaluator
    from autompc.metrics import RmseKstepMetric
    from autompc.graphs import KstepGrapher, InteractiveEvalGrapher

    metric = RmseKstepMetric(cartpole, k=50)
    #grapher = KstepGrapher(pendulum, kmax=50, kstep=5, evalstep=10)
    grapher = InteractiveEvalGrapher(cartpole)

    rng = np.random.default_rng(42)
    evaluator = HoldoutEvaluator(cartpole, trajs3[:500], metric, rng, holdout_prop=0.02,
            verbose=True) 
    #evaluator.add_grapher(grapher)
    #eval_score, _, graphs = evaluator(Model, s)
    #print("eval_score = {}".format(eval_score))
    #fig = plt.figure()
    #graph = graphs[0]
    #graph.set_obs_lower_bound("ang", -0.2)
    #graph.set_obs_upper_bound("ang", 0.2)
    #graphs[0](fig)
    #plt.show()
    #sys.exit(0)

    @memory.cache
    def run_tuner(runcount_limit=100, seed=42, n_jobs=1):
        tuner = ampc.ModelTuner(cartpole, evaluator)
        tuner.add_model(MLP)
        ret_value = tuner.run(rng=np.random.RandomState(seed), 
                runcount_limit=runcount_limit, n_jobs=n_jobs)
        return ret_value

    ret_value = run_tuner(runcount_limit=60)

    print(ret_value)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(range(len(ret_value["inc_costs"])), ret_value["inc_costs"])
    ax.set_title("Incumbent cost over time")
    ax.set_ylim([0.0, 5.0])
    ax.set_xlabel("Iterations.")
    ax.set_ylabel("Cost")
    plt.show()

    #from smac.scenario.scenario import Scenario
    #from smac.facade.smac_hpo_facade import SMAC4HPO
    #
    #scenario = Scenario({"run_obj": "quality",  
    #                     "runcount-limit": 50,  
    #                     "cs": cs,  
    #                     "deterministic": "true",
    #                     "n_jobs" : 10
    #                     })
    #
    #smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
    #        tae_runner=lambda cfg: evaluator(Model, cfg)[0])
    #
    #incumbent = smac.optimize()
    #
    #print("Done!")
    #
    #print(incumbent)

if __name__ == "__main__":
    main()
