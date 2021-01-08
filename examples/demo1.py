from pdb import set_trace
import sys, os
sys.path.append(os.getcwd() + "/..")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from joblib import Memory

from scipy.integrate import solve_ivp

memory = Memory("cache/")

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
    #y = np.copy(y)
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
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ctrl_text = ax.text(0.7, 0.95, '', transform=ax.transAxes)

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

dt = 0.01

umin = -2.0
umax = 2.0
udmax = 0.25

@memory.cache
def gen_trajs(traj_len=200, num_trajs=100, dt=dt):
    rng = np.random.default_rng(49)
    trajs = []
    for _ in range(num_trajs):
        theta0 = rng.uniform(-0.002, 0.002, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(cartpole, traj_len)
        for i in range(traj_len):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 1)
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

def gen_trajs_near_eq(traj_len=4, **kwargs):
    return gen_trajs(traj_len=traj_len, **kwargs)

memory = Memory("cache")




import autompc as ampc

cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
traj = ampc.zeros(cartpole, 10)
traj[2, "theta"] = 0.5
traj[2, "omega"] = 0.1
traj[2, "u"] = -0.2

print(traj[2].obs)
# Prints: [0.5 0.1 0.  0. ]
print(traj[2].ctrl)
# Prints: [-0.2] 


trajs = gen_trajs_near_eq(num_trajs=500)

from autompc.sysid import Koopman

cs = Koopman.get_configuration_space(cartpole)
print(cs)
# Prints:
#   Configuration space object:
#     Hyperparameters:
#       lasso_alpha_log10, Type: UniformFloat, Range: [-10.0, 2.0], Default: 0.0
#       method, Type: Categorical, Choices: {lstsq, lasso, stable}, Default: lstsq
#       poly_basis, Type: Categorical, Choices: {true, false}, Default: false
#       poly_degree, Type: UniformInteger, Range: [2, 8], Default: 3
#       product_terms, Type: Categorical, Choices: {false}, Default: false
#       trig_basis, Type: Categorical, Choices: {true, false}, Default: false
#       trig_freq, Type: UniformInteger, Range: [1, 8], Default: 1
#     Conditions:
#       lasso_alpha_log10 | method in {'lasso'}
#       poly_degree | poly_basis in {'true'}
#       trig_freq | trig_basis in {'true'}
cfg = cs.get_default_configuration()
cfg["method"] = "lstsq"

model = ampc.make_model(cartpole, Koopman, cfg)
model.train(trajs)



trajs2 = gen_trajs(num_trajs=500)
if False:
    from autompc.evaluators import FixedSetEvaluator
    from autompc.metrics import RmseKstepMetric
    from autompc.graphs import InteractiveEvalGrapher

    metric = RmseKstepMetric(cartpole, k=50)
    grapher = InteractiveEvalGrapher(cartpole)

    rng = np.random.default_rng(42)
    evaluator = FixedSetEvaluator(cartpole, trajs2[:50], metric, rng, 
            training_trajs=trajs) 
    evaluator.add_grapher(grapher)
    evaluator.add_grapher(grapher2)
    eval_score, _, graphs = evaluator(Koopman, cfg)
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
    plt.show()
    sys.exit(0)


from autompc.control import FiniteHorizonLQR
from autompc.tasks.task import Task
from autompc.tasks.quad_cost import QuadCost

task = Task(cartpole)
Q = np.diag([1.0, 1.0, 1.0, 1.0])
R = np.diag([0.01])
task.set_cost(QuadCost(cartpole, Q, R))

cs = FiniteHorizonLQR.get_configuration_space(cartpole, task, model)
cfg = cs.get_default_configuration()
cfg["horizon"] = 1000
con = ampc.make_controller(cartpole, task, model, FiniteHorizonLQR, cfg)

def run_sim(theta0):
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([theta0,0.0,0.0,0.0])
    sim_traj[0].obs[:] = x

    constate = con.traj_to_state(sim_traj[:1])
    for _ in range(1000):
        u, constate = con.run(constate, sim_traj[-1].obs)
        x = dt_cartpole_dynamics(x, u, dt)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
    return sim_traj

sim_traj = run_sim(theta0=0.4)

fig = plt.figure()
ax = fig.gca()
ax.set_aspect("equal")
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
#set_trace()
ani = animate_cartpole(fig, ax, dt, sim_traj)
ani.save("out/cartpole_test/aug27_01.mp4")
#plt.show()
