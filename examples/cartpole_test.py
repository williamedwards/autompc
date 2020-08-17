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

from autompc.sysid import ARX, Koopman, SINDy

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
    cfg["trig_basis"] = "false"
    cfg["poly_basis"] = "false"
    cfg["method"] = "lstsq"
    koop = ampc.make_model(cartpole, Koopman, cfg)
    koop.train(trajs)
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

# Test prediction

#traj = trajs[0]
#predobs, _ = koop.pred(traj[:10])

#koop_A, koop_B, state_func, cost_func = koop.to_linear()

#state = state_func(traj[:10])
#
#state = koop_A @ state + koop_B @ traj[10].ctrl
#state = koop_A @ state + koop_B @ traj[11].ctrl

#assert(np.allclose(state[-3:-1], traj[11].obs))

from autompc.sysid.dummy_linear import DummyLinear
A = np.array([[1.   , 0.01 , 0.   , 0.   ],
       [0.147, 0.995, 0.   , 0.   ],
       [0.   , 0.   , 1.   , 0.01 ],
       [0.098, 0.   , 0.   , 1.   ]])
B = np.array([[0.   ],
       [0.005],
       [0.   ],
       [0.01 ]])
#A =np.array([[ 0.9998053 ,  0.02499673,  0.        ,  0.        ],
#       [ 0.30379558,  0.98433411,  0.        , -0.01052765],
#       [ 0.        ,  0.        ,  0.99988456,  0.02499332],
#       [-0.18344387,  0.02941821,  0.01390835,  1.04997096]])
#B = np.array([[0.        ],
#       [0.        ],
#       [0.        ],
#       [0.02041931]])
class MyLinear(DummyLinear):
    def __init__(self, system):
        super().__init__(system, A, B)
#model = MyLinear(cartpole)
#set_trace()

if True:
    from autompc.evaluators import HoldoutEvaluator, FixedSetEvaluator
    from autompc.metrics import RmseKstepMetric
    from autompc.graphs import KstepGrapher, InteractiveEvalGrapher

    metric = RmseKstepMetric(cartpole, k=50)
    grapher = InteractiveEvalGrapher(cartpole)
    grapher2 = KstepGrapher(cartpole, kmax=50, kstep=5, evalstep=10)

    rng = np.random.default_rng(42)
    evaluator = FixedSetEvaluator(cartpole, trajs2, metric, rng, 
            training_trajs=trajs) 
    evaluator.add_grapher(grapher)
    evaluator.add_grapher(grapher2)
    #cs = Koopman.get_configuration_space(cartpole)
    #cfg = cs.get_default_configuration()
    #cfg["trig_basis"] = "false"
    #cfg["poly_basis"] = "false"
    #cfg["poly_degree"] = 3
    cs = MyLinear.get_configuration_space(cartpole)
    cfg = cs.get_default_configuration()
    eval_score, _, graphs = evaluator(MyLinear, cfg)
    print("eval_score = {}".format(eval_score))
    fig = plt.figure()
    #graph = graphs[0]
    #graph.set_obs_lower_bound("theta", -0.2)
    #graph.set_obs_upper_bound("theta", 0.2)
    #graph.set_obs_lower_bound("omega", -0.2)
    #graph.set_obs_upper_bound("omega", 0.2)
    #graph.set_obs_lower_bound("dx", -0.2)
    #graph.set_obs_upper_bound("dx", 0.2)
    #graph.set_obs_lower_bound("x", -0.2)
    #graph.set_obs_upper_bound("x", 0.2)
    graphs[1](fig)
    #plt.tight_layout()
    plt.show()
    sys.exit(0)


from autompc.control import InfiniteHorizonLQR, FiniteHorizonLQR, NonLinearMPC
#from autompc.control.mpc import LQRCost, LinearMPC
from cartpole_model import CartpoleModel

task = ampc.Task(cartpole)
Q = np.diag([1.0, 1.0, 1.0, 1.0])
R = np.diag([1.0])
#F = np.diag([10.0, 1.0, 1.0, 1.0])
task.set_quad_cost(Q, R)


if True:
    model = koop
    cs = FiniteHorizonLQR.get_configuration_space(cartpole, task, model)
    cfg = cs.get_default_configuration()
    cfg["horizon"] = 1000
    con = ampc.make_controller(cartpole, task, model, FiniteHorizonLQR, cfg)
else:
    model = CartpoleModel(cartpole)
    horizon = 0.8  # this is indeed too short for a frequency of 100 Hz model
    hh = 80
    con = NonLinearMPC(cartpole, model, task, horizon)
    con._guess = np.zeros(hh + (hh + 1) * 6) + 1e-5

@memory.cache
def run_sim(theta0):
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([theta0,0.0,0.0,0.0])
    sim_traj[0].obs[:] = x

    constate = con.traj_to_state(sim_traj[:1])
    for _ in range(1000):
        u, constate = con.run(constate, sim_traj[-1].obs)
        #u = np.zeros(1)
        x = dt_cartpole_dynamics(x, u, dt)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
    return sim_traj[-1].obs

theta0s = np.arange(-3, 3, 0.05)
data = np.zeros((25, len(theta0s)))
successes = []
for i, theta0 in enumerate(theta0s):
    final_obs = run_sim(theta0)
    if abs(final_obs[0]) < 0.01:
        print("Theta0 {} is success!".format(theta0))
        successes.append(1)
        data[:,i] = 1
    else:
        print("Theta0 {} is a failure!".format(theta0))
        successes.append(0)
print(len(theta0s))
fig = plt.figure()
ax = fig.gca()
ax.set_aspect("equal")
ax.imshow(data, extent=[theta0s[0], theta0s[-1], 0.0, 1.0], cmap="RdYlGn")
ax.set_xlabel("Starting Angle")
plt.show()


#fig = plt.figure()
#ax = fig.gca()
#ax.set_aspect("equal")
#ax.set_xlim([-1.1, 1.1])
#ax.set_ylim([-1.1, 1.1])
##set_trace()
#ani = animate_cartpole(fig, ax, dt, sim_traj)
#ani.save("out/cartpole_test/aug11_01.mp4")
##plt.show()
