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

planar_drone = ampc.System(["x", "dx", "y", "dy", "theta", "omega"], ["u1", "u2"])

def planar_drone_dynamics(y, u, g, m, r, I):
    """
    Parameters
    ----------
        y : states
        u : control

    Returns
    -------
        A list describing the dynamics of the cart cart pole
    """
    x, dx, y, dy, theta, omega = y
    u1, u2  = u
    return [dx,
            -(u1 + u2) * np.sin(theta),
            dy,
            (u1 + u2) * np.cos(theta) - m * g,
            omega,
            r / I * (u1 - u2)]

def dt_planar_drone_dynamics(y,u,dt,g=0.0,m=1,r=1.0,I=1.0):
    y = np.copy(y)
    sol = solve_ivp(lambda t, y: planar_drone_dynamics(y, u, g, m, r, I), (0, dt), y, 
            t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y = sol.y.reshape((6,))
    return y

def animate_planar_drone(fig, ax, dt, traj, r=1.0):
    ax.grid()

    line, = ax.plot([-1.0, 0.0], [1.0, 0.0], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ctrl_text = ax.text(0.7, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([0.0, 0.0], [0.0, -1.0])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        #i = min(i, ts.shape[0])
        x1 = traj[i, "x"] + np.cos(traj[i, "theta"])
        y1 = traj[i, "y"] + np.sin(traj[i, "theta"])
        x2 = traj[i, "x"] - np.cos(traj[i, "theta"])
        y2 = traj[i, "y"] - np.sin(traj[i, "theta"])
        line.set_data([x1, x2], [y1, y2])
        time_text.set_text('t={:.2f}'.format(dt*i))
        ctrl_text.set_text("u1={:.2f}, u2={:.2f}".format(traj[i,"u1"], traj[i,"u2"]))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani

dt = 0.01

umin = -2.0
umax = 2.0
udmax = 0.25

# Generate trajectories for training
num_trajs = 500

@memory.cache
def gen_trajs(num_trajs=num_trajs):
    rng = np.random.default_rng(49)
    trajs = []
    for _ in range(num_trajs):
        theta0 = rng.uniform(-0.02, 0.02, 1)[0]
        x0 = rng.uniform(-0.02, 0.02, 1)[0]
        y0 = rng.uniform(-0.02, 0.02, 1)[0]
        y = [x0, 0.0, y0, 0.0, theta0, 0.0]
        traj = ampc.zeros(planar_drone, 5)
        for i in range(5):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 2)
            y = dt_planar_drone_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs
trajs = gen_trajs()

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
    cs = Koopman.get_configuration_space(planar_drone)
    cfg = cs.get_default_configuration()
    cfg["trig_basis"] = "false"
    cfg["poly_basis"] = "false"
    koop = ampc.make_model(planar_drone, Koopman, cfg)
    koop.train(trajs)
    return koop

def train_sindy():
    cs = SINDy.get_configuration_space(planar_drone)
    cfg = cs.get_default_configuration()
    cfg["trig_basis"] = "true"
    cfg["poly_basis"] = "false"
    sindy = ampc.make_model(planar_drone, SINDy, cfg)
    sindy.train(trajs)
    return sindy

#koop = train_koop()
sindy = train_sindy()

# Test prediction

#model = koop
model = sindy
set_trace()

if False:
    from autompc.evaluators import HoldoutEvaluator
    from autompc.metrics import RmseKstepMetric
    from autompc.graphs import KstepGrapher, InteractiveEvalGrapher

    metric = RmseKstepMetric(planar_drone, k=50)
    #grapher = KstepGrapher(pendulum, kmax=50, kstep=5, evalstep=10)
    grapher = InteractiveEvalGrapher(planar_drone)

    rng = np.random.default_rng(42)
    evaluator = HoldoutEvaluator(planar_drone, trajs, metric, rng, holdout_prop=0.25) 
    evaluator.add_grapher(grapher)
    #cs = Koopman.get_configuration_space(planar_drone)
    #cfg = cs.get_default_configuration()
    #cfg["trig_basis"] = "false"
    #cfg["poly_basis"] = "false"
    #cfg["poly_degree"] = 3
    cs = MyLinear.get_configuration_space(planar_drone)
    cfg = cs.get_default_configuration()
    eval_score, _, graphs = evaluator(MyLinear, cfg)
    print("eval_score = {}".format(eval_score))
    fig = plt.figure()
    graph = graphs[0]
    graph.set_obs_lower_bound("theta", -0.2)
    graph.set_obs_upper_bound("theta", 0.2)
    graph.set_obs_lower_bound("omega", -0.2)
    graph.set_obs_upper_bound("omega", 0.2)
    graph.set_obs_lower_bound("dx", -0.2)
    graph.set_obs_upper_bound("dx", 0.2)
    graphs[0](fig)
    #plt.tight_layout()
    plt.show()


from autompc.control import InfiniteHorizonLQR, FiniteHorizonLQR
#from autompc.control.mpc import LQRCost, LinearMPC

task = ampc.Task(planar_drone)
Q = np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 1.0])
R = np.diag([0.1, 0.1])
task.set_quad_cost(Q, R)


cs = FiniteHorizonLQR.get_configuration_space(planar_drone, task, model)
cfg = cs.get_default_configuration()
cfg["horizon"] = 100
con = ampc.make_controller(planar_drone, task, model, FiniteHorizonLQR, cfg)

sim_traj = ampc.zeros(planar_drone, 1)
x = np.array([0.0,0.0,1.0,0.0,1.0,0.0])
sim_traj[0].obs[:] = x
set_trace()

constate = con.traj_to_state(sim_traj[:1])
#K2 = -np.array([[103.95635975,  27.79511119,  -2.82043609,  -4.75267206]])
#X1 = np.array([[5053.06324338, -268.47827486,    0.,            0.,        ],
# [-268.47827486, 5054.34910433,    0.,            0.,        ],
# [   0.,            0.,           10.,            0.,        ],
# [   0.,            0.,            0.,           11.,        ]])
#X2 = np.array([[183594.97608883,  51609.12730918, -11131.75586988,
#         -14457.5851753 ],
#        [ 51609.12730918,  14625.69494522,  -3284.68327891,
#          -4266.9610752 ],
#        [-11131.75586988,  -3284.68327891,   1654.86316438,
#           1311.0117305 ],
#        [-14457.5851753 ,  -4266.9610752 ,   1311.0117305 ,
#           1588.48817528]])
#N = np.zeros((4,1))
#set_trace()
for _ in range(1000):
    u, constate = con.run(constate, sim_traj[-1].obs)
    x = dt_planar_drone_dynamics(x, u, dt)
    sim_traj[-1].ctrl[:] = u
    sim_traj = ampc.extend(sim_traj, [x], [[0.0, 0.0]])

#plt.plot(sim_traj[:,"x1"], sim_traj[:,"x2"], "b-o")
#plt.show()
print("K:")
print(con.K)

fig = plt.figure()
ax = fig.gca()
ax.set_aspect("equal")
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
#set_trace()
ani = animate_planar_drone(fig, ax, dt, sim_traj)
#ani.save("out/cartpole_test/aug05_05.mp4")
plt.show()
