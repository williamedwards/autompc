"""
Test the nmpc.py file.
"""
import argparse
import numpy as np
import autompc as ampc
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pdb import set_trace

from autompc.sysid.dummy_nonlinear import DummyNonlinear
from autompc.control import NonLinearMPC, LinearMPC
from autompc import Task

from joblib import Memory
from scipy.integrate import solve_ivp


def animate_pendulum(fig, ax, dt, traj):
    ax.grid()

    line, = ax.plot([0.0, 0.0], [0.0, -1.0], 'o-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([0.0, 0.0], [0.0, -1.0])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        #i = min(i, ts.shape[0])
        line.set_data([0.0, np.sin(traj[i,"ang"]+np.pi)], 
                [0.0, -np.cos(traj[i,"ang"] + np.pi)])
        time_text.set_text('t={:2f}'.format(dt*i))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani

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
        ctrl_text.set_text("u={:.3e}".format(traj[i,"u"]))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani


def animate_planar_drone(fig, ax, dt, traj, r=0.25):
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
        x1 = traj[i, "x"] + r*np.cos(traj[i, "theta"])
        y1 = traj[i, "y"] + r*np.sin(traj[i, "theta"])
        x2 = traj[i, "x"] - r*np.cos(traj[i, "theta"])
        y2 = traj[i, "y"] - r*np.sin(traj[i, "theta"])
        line.set_data([x1, x2], [y1, y2])
        time_text.set_text('t={:.2f}'.format(dt*i))
        ctrl_text.set_text("u1={:.2f}, u2={:.2f}".format(traj[i,"u1"], traj[i,"u2"]))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, frames=traj.size, interval=dt*1000,
            blit=False, init_func=init, repeat_delay=1000)

    return ani

def test_dummy():
    dummy_sys = ampc.System(["x1", "x2"], ["u"])
    dummy_model = DummyNonlinear(dummy_sys)

    task1 = Task(dummy_sys)
    Q = np.eye(2)
    R = np.eye(1)
    F = 10 * np.eye(2)
    task1.set_quad_cost(Q, R, F)

    nmpc = NonLinearMPC(dummy_sys, dummy_model, task1)
    # just give a random initial state
    sim_traj = ampc.zeros(dummy_sys, 1)
    x = np.array([2, 1.0])
    sim_traj[0].obs[:] = x

    for _ in range(10):
        u, _ = nmpc.run(sim_traj)
        print('u = ', u)
        x = dummy_model.traj_to_state(sim_traj)
        x = dummy_model.pred(x, u)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])

    print(sim_traj[:, "x1"], sim_traj[:, 'x2'])
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.plot(sim_traj[:, 'x1'], sim_traj[:, 'x2'])
    plt.show()


# Here I test the discrete sindy example (maybe continuous is better to be honest...)
def test_sindy_pendulum():
    from autompc.sysid import SINDy
    cs = SINDy.get_configuration_space(pendulum)
    s = cs.get_default_configuration()
    s["trig_basis"] = "true"
    s["trig_freq"] = 1
    s["poly_basis"] = "false"
    dt = 0.025
    pendulum.dt = dt
    umin, umax = -2, 2
    trajs = collect_pendulum_trajs(dt, umin, umax)
    model = ampc.make_model(pendulum, SINDy, s)
    model.train(trajs)
    # Now it's time to apply the controller
    task1 = Task(pendulum)
    Q = np.diag([100.0, 1.0])
    R = np.diag([1.0]) 
    F = 100 * np.eye(2)
    task1.set_quad_cost(Q, R, F)

    horizon = 1.0   # this is indeed too short for a frequency of 100 Hz model
    nmpc = NonLinearMPC(pendulum, model, task1, horizon)
    # just give a random initial state
    sim_traj = ampc.zeros(pendulum, 1)
    x = np.array([-np.pi, 0])
    sim_traj[0].obs[:] = x

    constate = nmpc.traj_to_state(sim_traj[:1])
    for _ in range(400):
        u, constate = nmpc.run(constate, sim_traj[-1].obs)
        print('u = ', u)
        x = model.traj_to_state(sim_traj)
        #x = model.pred(x, u)
        x = dt_pendulum_dynamics(x, u, dt)
        sim_traj[-1, "torque"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])

    print(sim_traj[:, "ang"], sim_traj[:, 'angvel'])
    #fig = plt.figure()
    #ax = fig.gca()
    #ax.set_aspect("equal")
    #ax.plot(sim_traj[:, 'ang'], sim_traj[:, 'angvel'])
    #plt.show()
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ani = animate_pendulum(fig, ax, dt, sim_traj)
    plt.show()

def test_sindy_planar_drone():
    from autompc.sysid import SINDy
    from planar_drone_model import PlanarDroneModel
    import time
    cs = SINDy.get_configuration_space(planar_drone)
    s = cs.get_default_configuration()
    s["trig_basis"] = "true"
    s["trig_freq"] = 1
    s["poly_basis"] = "false"
    dt = 0.025
    umin, umax = -2, 2
    planar_drone.dt = dt
    trajs = collect_planar_drone_trajs(dt, umin, umax)
    #model = ampc.make_model(planar_drone, SINDy, s)
    #model.train(trajs)
    model = PlanarDroneModel(planar_drone)
    #set_trace()
    #time.sleep(1.0)


    # Evaluate model
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
        eval_score, _, graphs = evaluator(SINDy, s)
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


    # Now it's time to apply the controller
    task1 = Task(planar_drone)
    Q = np.diag([100.0, 0.0, 10.0, 0.0, 10.0, 0.0])
    R = np.diag([0.01, 0.01]) 
    F = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    task1.set_quad_cost(Q, R, F)

    horizon = 0.25  # this is indeed too short for a frequency of 100 Hz model
    hh = 10
    nmpc = NonLinearMPC(planar_drone, model, task1, horizon)
    nmpc._guess = np.zeros(hh * 2 + (hh + 1) * 6) + 1e-5
    # just give a random initial state
    sim_traj = ampc.zeros(planar_drone, 1)
    x = np.array([0.0, 0, 1.0, 0, 0.5, 0])
    sim_traj[0].obs[:] = x
    us = []

    constate = nmpc.traj_to_state(sim_traj[:1])
    for step in range(400):
        u, constate = nmpc.run(constate, sim_traj[-1].obs)
        #u = -np.zeros((1,))
        print('u = ', u)
        # x = model.pred(x, u)
        x = dt_planar_drone_dynamics(x, u, dt)
        sim_traj[-1].ctrl[:] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0,0.0]])

        us.append(u)
    # print(np.array(us))
    # raise

    #print(sim_traj[:, "ang"], sim_traj[:, 'angvel'])
    #fig = plt.figure()
    #ax = fig.gca()
    #ax.set_aspect("equal")
    #ax.plot(sim_traj[:, 'ang'], sim_traj[:, 'angvel'])
    #plt.show()
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ani = animate_planar_drone(fig, ax, dt, sim_traj)
    plt.show()
    #ani.save("out/nmpc_test/aug05_04.mp4")

def test_true_dyn_planar_drone():
    from autompc.sysid import SINDy
    from planar_drone_model import PlanarDroneModel


    dt = 0.025
    planar_drone.dt = dt
    umin, umax = -2, 2
    trajs = collect_planar_drone_trajs(dt, umin, umax)
    model = PlanarDroneModel(planar_drone)

    # Now it's time to apply the controller
    task1 = Task(planar_drone)
    Q = np.diag([100.0, 0.0, 10.0, 0.0, 10.0, 0.0])
    R = np.diag([0.01, 0.01]) 
    F = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    task1.set_quad_cost(Q, R, F)

    horizon = 0.25  # this is indeed too short for a frequency of 100 Hz model
    hh = 10
    nmpc = NonLinearMPC(planar_drone, model, task1, horizon)
    nmpc._guess = np.zeros(hh * 2 + (hh + 1) * 6) + 1e-5
    # just give a random initial state
    sim_traj = ampc.zeros(planar_drone, 1)
    x = np.array([0.0, 0, 1.0, 0, 0.5, 0])
    sim_traj[0].obs[:] = x
    us = []

    constate = nmpc.traj_to_state(sim_traj[:1])
    for step in range(400):
        u, constate = nmpc.run(constate, sim_traj[-1].obs)
        #u = -np.zeros((1,))
        print('u = ', u)
        # x = model.pred(x, u)
        x = dt_planar_drone_dynamics(x, u, dt)
        sim_traj[-1].ctrl[:] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0,0.0]])

        us.append(u)
    # print(np.array(us))
    # raise

    #print(sim_traj[:, "ang"], sim_traj[:, 'angvel'])
    #fig = plt.figure()
    #ax = fig.gca()
    #ax.set_aspect("equal")
    #ax.plot(sim_traj[:, 'ang'], sim_traj[:, 'angvel'])
    #plt.show()
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ani = animate_planar_drone(fig, ax, dt, sim_traj)
    plt.show()
    #ani.save("out/nmpc_test/aug05_04.mp4")

def test_sindy_cartpole():
    from autompc.sysid import SINDy
    from cartpole_model import CartpoleModel
    #cs = CartpoleModel.get_configuration_space(cartpole)
    cs = SINDy.get_configuration_space(cartpole)
    s = cs.get_default_configuration()
    s["trig_basis"] = "true"
    #s["trig_freq"] = 1
    s["poly_basis"] = "false"
    dt = 0.01
    umin, umax = -2, 2
    trajs = collect_cartpole_trajs(dt, umin, umax)
    cartpole.dt = dt
    model = ampc.make_model(cartpole, SINDy, s)
    model.trig_interaction = True
    model.train(trajs)

    # Evaluate model
    if True:
        from autompc.evaluators import HoldoutEvaluator
        from autompc.metrics import RmseKstepMetric
        from autompc.graphs import KstepGrapher, InteractiveEvalGrapher

        metric = RmseKstepMetric(cartpole, k=50)
        #grapher = KstepGrapher(pendulum, kmax=50, kstep=5, evalstep=10)
        grapher = InteractiveEvalGrapher(cartpole)

        rng = np.random.default_rng(42)
        evaluator = HoldoutEvaluator(cartpole, trajs, metric, rng, holdout_prop=0.25) 
        evaluator.add_grapher(grapher)
        eval_score, _, graphs = evaluator(SINDy, s)
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


    # Now it's time to apply the controller
    task1 = Task(cartpole)
    Q = np.diag([10.0, 1.0, 10.0, 1.0])
    R = np.diag([1.0]) 
    F = np.diag([10., 10., 2., 10.])
    task1.set_quad_cost(Q, R, F)

    horizon = 0.1  # this is indeed too short for a frequency of 100 Hz model
    hh = 10
    nmpc = NonLinearMPC(cartpole, model, task1, horizon)
    nmpc._guess = np.zeros(hh + (hh + 1) * 4) + 1e-5
    # just give a random initial state
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([0.0, 0, 0, 0])
    sim_traj[0].obs[:] = x
    us = []

    constate = nmpc.traj_to_state(sim_traj[:1])
    for step in range(400):
        u, constate = nmpc.run(constate, sim_traj[-1].obs)
        #u = -np.zeros((1,))
        print('u = ', u)
        # x = model.pred(x, u)
        x = dt_cartpole_dynamics(x, u, dt)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        # compare what happens if zero control for 80 steps
        if False and np.abs(u) > 1e-2:
            states = [saved_state]
            for _ in range(nmpc.horizon):
                states.append(dt_cartpole_dynamics(states[-1].copy(), 0, dt))
            fig, ax = plt.subplots(2, 2)
            ax = ax.reshape(-1)
            pred = nmpc._guess[:4 * (hh + 1)].reshape((-1, 4))
            states = np.array(states)
            for j in range(4):
                ax[j].plot(states[:, j], label='zero control')
                ax[j].plot(pred[:, j], label='nmpc')
                ax[j].legend()
            plt.show()

        us.append(u)
    # print(np.array(us))
    # raise

    #print(sim_traj[:, "ang"], sim_traj[:, 'angvel'])
    #fig = plt.figure()
    #ax = fig.gca()
    #ax.set_aspect("equal")
    #ax.plot(sim_traj[:, 'ang'], sim_traj[:, 'angvel'])
    #plt.show()
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ani = animate_cartpole(fig, ax, dt, sim_traj)
    plt.show()
    #ani.save("out/nmpc_test/aug05_04.mp4")

def test_true_dyn_cartpole():
    from ilqr_test import CartPole, cartpole
    dt = cartpole.dt
    umin, umax = -15, 15
    model = CartPole()

    # Now it's time to apply the controller
    task1 = Task(cartpole)
    Q = np.diag([1., 1., 1., 1.])
    R = np.diag([0.1]) 
    F = np.diag([10.0, 10.0, 10., 10.]) * 10
    task1.set_quad_cost(Q, R, F)
    task1.set_ctrl_bounds(umin, umax)

    init_state = np.array([np.pi, 0.0, 0., 0.])
    horizon_int = 40

    horizon = 2.0  # this is indeed too short for a frequency of 100 Hz model
    hh = 40
    nmpc = NonLinearMPC(cartpole, model, task1, horizon)
    nmpc._guess = np.random.normal(size=hh + (hh + 1) * 4)
    # just give a random initial state
    sim_traj = ampc.zeros(cartpole, 1)
    sim_traj[0].obs[:] = init_state
    us = []

    constate = nmpc.traj_to_state(sim_traj[:1])
    for step in range(400):
        state = sim_traj[-1].obs
        u, constate = nmpc.run(constate, sim_traj[-1].obs)
        if np.linalg.norm(state) < 1e-3:
            break
        #u = -np.zeros((1,))
        print('u = ', u)
        # x = model.pred(x, u)
        x = model.pred(state, u)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        # compare what happens if zero control for 80 steps
        us.append(u)
    
    fig, ax = plt.subplots(3, 2)
    ax = ax.reshape(-1)
    state_names = ['theta', 'omega', 'x', 'dx']
    times = np.arange(sim_traj.obs.shape[0]) * cartpole.dt
    for i in range(4):
        ax[i].plot(times, sim_traj.obs[:, i])
        ax[i].set_ylabel(state_names[i])
        ax[i].set_xlabel('Time [s]')
    ax[4].plot(times[:-1], sim_traj.ctrls[:-1])
    ax[4].set(xlabel='Time [s]', ylabel='u')
    fig.tight_layout()
    fig.savefig('cartpole_nmpc_state_bound_control.png')

def test_linear_cartpole():
    from autompc.sysid.dummy_linear import DummyLinear
    A = np.array([[1.   , 0.01 , 0.   , 0.   ],
           [0.147, 0.995, 0.   , 0.   ],
           [0.   , 0.   , 1.   , 0.01 ],
           [0.098, 0.   , 0.   , 1.   ]])
    B = np.array([[0.   ],
           [0.005],
           [0.   ],
           [0.01 ]])

    class MyLinear(DummyLinear):
        def __init__(self, system):
            super().__init__(system, A, B)
            self.dt = 0.01

    model = MyLinear(cartpole)

    dt = 0.01
    cartpole.dt = dt

    # Create task
    task1 = Task(cartpole)
    Q = np.diag([10.0, 1.0, 10.0, 1.0])
    R = np.diag([0.01]) 
    F = np.diag([10., 10., 10., 10.]) * 100
    # F = np.zeros((4,4))
    task1.set_quad_cost(Q, R, F)

    # Initialize controller
    horizon = 0.1  # this is indeed too short for a frequency of 100 Hz model
    hh = 10
    nmpc = NonLinearMPC(cartpole, model, task1, horizon)
    nmpc._guess = np.zeros(hh + (hh + 1) * 4) + 1e-5
    # nmpc = LinearMPC(cartpole, model, task1, hh)
    # just give a random initial state
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([0.1, 0, 0, 0])
    sim_traj[0].obs[:] = x
    us = []

    # Run simulation
    constate = nmpc.traj_to_state(sim_traj[:1])
    for step in range(500):
        u, constate = nmpc.run(constate, sim_traj[-1].obs)
        #u = -np.zeros((1,))
        # x = model.pred(x, u)
        print('u = ', u, 'x = ', x)
        x = dt_cartpole_dynamics(x, u, dt)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])
        # compare what happens if zero control for 80 steps
        us.append(u)
    # Siplay results
    fig = plt.figure()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ani = animate_cartpole(fig, ax, dt, sim_traj)
    plt.show()


memory = Memory("cache")

pendulum = ampc.System(["ang", "angvel"], ["torque"])
cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
planar_drone = ampc.System(["x", "dx", "y", "dy", "theta", "omega"], ["u1", "u2"])

def pendulum_dynamics(y,u,g=9.8,m=1,L=1,b=0.1):
    theta, omega = y
    return [omega,((u[0] - b*omega)/(m*L**2)
        - g*np.sin(theta)/L)]

def dt_pendulum_dynamics(y,u,dt,g=9.8,m=1,L=1,b=0.1):
    y[0] += np.pi
    sol = solve_ivp(lambda t, y: pendulum_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y = sol.y.reshape((2,))
    y[0] -= np.pi
    return sol.y.reshape((2,))

def collect_pendulum_trajs(dt, umin, umax):
    # Generate trajectories for training
    num_trajs = 100
    trajs = gen_pendulum_trajs(dt, num_trajs, umin, umax)
    return trajs


@memory.cache
def gen_pendulum_trajs(dt, num_trajs, umin, umax):
    rng = np.random.default_rng(42)
    trajs = []
    for _ in range(num_trajs):
        y = [-np.pi, 0.0]
        traj = ampc.zeros(pendulum, 400)
        for i in range(400):
            traj[i].obs[:] = y
            u = rng.uniform(umin, umax, 1)
            y = dt_pendulum_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

def cartpole_dynamics(y, u, g = 9.8, m_c = 1, m_p = 1.00, L = 1, b = 1.00):
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
    #return [omega,
    #        g * np.sin(theta)/L - b * omega / (m*L**2) + u * np.sin(theta)/L,
    #        dx,
    #        u]
    return np.array([omega,
            1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                - (m_c+m_p+m_p)*g*np.sin(theta)
                - b*omega),
            dx,
            1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                (L*omega**2 + g*np.cos(theta)))])

def dt_cartpole_dynamics(y,u,dt):
    y = np.copy(np.array(y))
    y[0] += np.pi
    #sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    #if not sol.success:
    #    raise Exception("Integration failed due to {}".format(sol.message))
    #y = sol.y.reshape((4,))
    y = y + dt * cartpole_dynamics(y, u[0])
    y[0] -= np.pi
    return y

def collect_cartpole_trajs(dt, umin, umax):
    # Generate trajectories for training
    num_trajs = 100
    trajs = gen_cartpole_trajs(dt, num_trajs, umin, umax)
    return trajs


@memory.cache
def gen_cartpole_trajs(dt, num_trajs, umin, umax):
    rng = np.random.default_rng(49)
    trajs = []
    for _ in range(num_trajs):
        theta0 = rng.uniform(-0.002, 0.002, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(cartpole, 200)
        for i in range(200):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 1)
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs


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
    x, dx, y2, dy, theta, omega = y
    u1, u2  = u
    return [dx,
            -(u1 + u2) * np.sin(theta),
            dy,
            (u1 + u2) * np.cos(theta) - m * g,
            omega,
            r / I * (u1 - u2)]

def dt_planar_drone_dynamics(y,u,dt,g=0.0,m=1,r=0.25,I=1.0):
    y = np.copy(y)
    sol = solve_ivp(lambda t, y: planar_drone_dynamics(y, u, g, m, r, I), (0, dt), y, 
            t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y = sol.y.reshape((6,))
    #y += dt * np.array(planar_drone_dynamics(y, u, g, m, r, I))
    return y

#@memory.cache
def gen_planar_drone_trajs(dt, num_trajs, umin, umax):
    print("Generating planar drone trajs.")
    rng = np.random.default_rng(49)
    trajs = []
    for _ in range(num_trajs):
        theta0 = rng.uniform(-0.02, 0.02, 1)[0]
        x0 = rng.uniform(-0.02, 0.02, 1)[0]
        y0 = rng.uniform(-0.02, 0.02, 1)[0]
        y = [x0, 0.0, y0, 0.0, theta0, 0.0]
        traj = ampc.zeros(planar_drone, 200)
        for i in range(200):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 2)
            y = dt_planar_drone_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

def collect_planar_drone_trajs(dt, umin, umax):
    # Generate trajectories for training
    num_trajs = 100
    trajs = gen_planar_drone_trajs(dt, num_trajs, umin, umax)
    return trajs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['koopman', 'sindy-pendulum', 
        "sindy-cartpole", "linear-cartpole", "sindy-planar-drone", 
        "true-dyn-cartpole", "true-dyn-planar-drone"], 
        default='sindy-cartpole', help='Specify which system id to test')
    args = parser.parse_args()
    if args.model == 'koopman':
        test_dummy()
    if args.model == 'sindy-pendulum':
        test_sindy_pendulum()
    if args.model == 'sindy-cartpole':
        test_sindy_cartpole()
    if args.model == 'linear-cartpole':
        test_linear_cartpole()
    if args.model == "sindy-planar-drone":
        test_sindy_planar_drone()
    if args.model == "true-dyn-cartpole":
        test_true_dyn_cartpole()
    if args.model == "true-dyn-planar-drone":
        test_true_dyn_planar_drone()
