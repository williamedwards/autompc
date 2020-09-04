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
    y[0] += np.pi
    sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y = sol.y.reshape((4,))
    #y += dt * cartpole_simp_dynamics(y,u[0],g,m,L,b)
    y[0] -= np.pi
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
trajs2 = gen_trajs(200)

from autompc.sysid import ARX, Koopman, SINDy, GaussianProcess

def train_gp(datasize=50, num_trajs=1):
    cs = GaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    gp = ampc.make_model(cartpole, GaussianProcess, cfg)
    mytrajs = [trajs2[i][:datasize] for i in range(num_trajs)]
    gp.train(mytrajs)
    return gp

@memory.cache
def train_arx(k=2):
    cs = ARX.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    cfg["history"] = k
    arx = ampc.make_model(cartpole, ARX, cfg)
    arx.train(trajs)
    return arx

#@memory.cache
def train_koop():
    cs = Koopman.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    cfg["trig_basis"] = "true"
    cfg["poly_basis"] = "false"
    cfg["method"] = "lstsq"
    koop = ampc.make_model(cartpole, Koopman, cfg)
    koop.train(trajs2[:50])
    return koop

def train_sindy():
    cs = SINDy.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    cfg["trig_basis"] = "true"
    cfg["trig_freq"] = 1
    cfg["poly_basis"] = "false"
    sindy = ampc.make_model(cartpole, SINDy, cfg)
    sindy.train(trajs2)
    return sindy

arx = train_arx(k=4)
koop = train_koop()
sindy = train_sindy()

## Training GP model
#import timeit
#sizes = []
#times = []
#for datasize in range(10, 200, 10):
#    num_trajs = 10
#    gp = train_gp(datasize=datasize, num_trajs=num_trajs)
#    t = timeit.Timer(lambda: gp.pred(np.zeros(4,), np.ones(1,)))
#    print(t.timeit(number=1))
#    rep = t.repeat(number=3)
#    print(rep)
#    sizes.append(datasize*num_trajs)
#    times.append(rep[-1]*1000.0)
#plt.figure()
#ax = plt.gca()
#ax.plot(sizes, times)
#ax.set_xlabel("Training data points.")
#ax.set_ylabel("Inference time (ms)")
#plt.show()
#sys.exit(0)

if True:
    from autompc.evaluators import HoldoutEvaluator, FixedSetEvaluator
    from autompc.metrics import RmseKstepMetric
    from autompc.graphs import KstepGrapher, InteractiveEvalGrapher

    metric = RmseKstepMetric(cartpole, k=10)
    grapher = InteractiveEvalGrapher(cartpole)
    grapher2 = KstepGrapher(cartpole, kmax=50, kstep=5, evalstep=10)

    rng = np.random.default_rng(42)
    evaluator = FixedSetEvaluator(cartpole, trajs2[1:0], metric, rng, 
            training_trajs=[trajs2[0][:100], trajs2[3][150:200]]) 
    evaluator.add_grapher(grapher)
    #evaluator.add_grapher(grapher2)
    cs = GaussianProcess.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    #cfg["trig_basis"] = "true"
    #cfg["poly_basis"] = "false"
    #cfg["poly_degree"] = 3
    #cs = MyLinear.get_configuration_space(cartpole)
    #cfg = cs.get_default_configuration()
    eval_score, _, graphs = evaluator(GaussianProcess, cfg)
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
    #plt.tight_layout()
    plt.show()
    sys.exit(0)


from autompc.control import InfiniteHorizonLQR, FiniteHorizonLQR, NonLinearMPC
#from autompc.control.mpc import LQRCost, LinearMPC
from cartpole_model import CartpoleModel
from autompc.tasks.task import Task
from autompc.tasks.quad_cost import QuadCost

task = Task(cartpole)
Q = np.diag([1.0, 1.0, 1.0, 1.0])
R = np.diag([1.0])
#F = np.diag([10.0, 1.0, 1.0, 1.0])
task.set_cost(QuadCost(cartpole, Q, R))

model = koop
cs = FiniteHorizonLQR.get_configuration_space(cartpole, task, model)
cfg = cs.get_default_configuration()
cfg["horizon"] = 1000
con = ampc.make_controller(cartpole, task, model, FiniteHorizonLQR, cfg)

def run_sim(theta0, break_sim=True):
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([theta0,0.0,0.0,0.0])
    sim_traj[0].obs[:] = x

    constate = con.traj_to_state(sim_traj[:1])
    Q, _, _ = task.get_cost().get_cost_matrices()
    for _ in range(1000):
        u, constate = con.run(constate, sim_traj[-1].obs)
        #u = np.zeros(1)
        x = dt_cartpole_dynamics(x, u, dt)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        state_cost = x.T @ Q @ x
        if break_sim and state_cost > 10000.0:
            break
    return sim_traj

theta0_lower = 0.0
theta0_upper = 3.0
while theta0_upper - theta0_lower > 0.01:
    theta0 = (theta0_upper + theta0_lower) / 2
    print(f"{theta0=}")
    sim_traj = run_sim(theta0)
    success = (abs(sim_traj[-1, "theta"]) < 0.01
            and abs(sim_traj[-1, "x"])  < 0.01)
    if success:
        theta0_lower = theta0
    else:
        theta0_upper = theta0

print(f"Max Theta = {theta0_lower}")
sim_traj = run_sim(theta0_lower, break_sim=True)

fig = plt.figure()
ax = fig.gca()
ax.set_aspect("equal")
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
#set_trace()
ani = animate_cartpole(fig, ax, dt, sim_traj)
#ani.save("out/cartpole_test/aug11_01.mp4")
plt.show()
