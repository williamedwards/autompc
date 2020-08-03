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
from autompc.control import NonLinearMPC
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

def test_dummy():
    dummy_sys = ampc.System(["x1", "x2"], ["u"])
    dummy_model = DummyNonlinear(dummy_sys)

    task1 = Task(dummy_sys)
    Q = np.eye(2)
    R = np.eye(1)
    F = 10 * np.eye(2)
    task1.set_quad_cost(Q, R, F)

    horizon = 8
    nmpc = NonLinearMPC(dummy_sys, dummy_model, task1, horizon)
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
    umin, umax = -2, 2
    trajs = collect_pendulum_trajs(dt, umin, umax)
    model = ampc.make_model(pendulum, SINDy, s)
    model.train(trajs)
    # Now it's time to apply the controller
    task1 = Task(pendulum)
    Q = np.diag([100.0, 1.0])
    R = np.diag([1.0]) 
    F = 10 * np.eye(2)
    task1.set_quad_cost(Q, R, F)

    horizon = 40  # this is indeed too short for a frequency of 100 Hz model
    nmpc = NonLinearMPC(pendulum, model, task1, horizon)
    # just give a random initial state
    sim_traj = ampc.zeros(pendulum, 1)
    x = np.array([-np.pi, 0])
    sim_traj[0].obs[:] = x

    for _ in range(400):
        u, _ = nmpc.run(sim_traj)
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

def test_sindy_cartpole():
    from autompc.sysid import SINDy
    cs = SINDy.get_configuration_space(cartpole)
    s = cs.get_default_configuration()
    s["trig_basis"] = "true"
    s["trig_freq"] = 1
    s["poly_basis"] = "false"
    dt = 0.025
    umin, umax = -2, 2
    trajs = collect_cartpole_trajs(dt, umin, umax)
    model = ampc.make_model(cartpole, SINDy, s)
    model.train(trajs)
    #set_trace()

    # Evaluate model
    if False:
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
    Q = np.diag([1.0, 1.0, 1.0, 1.0])
    R = np.diag([1.0]) 
    F = np.eye(4)
    task1.set_quad_cost(Q, R, F)

    horizon = 80  # this is indeed too short for a frequency of 100 Hz model
    nmpc = NonLinearMPC(cartpole, model, task1, horizon)
    # just give a random initial state
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.array([0.0, 0, 0, 0])
    sim_traj[0].obs[:] = x

    for _ in range(400):
        u, _ = nmpc.run(sim_traj)
        #u = -np.ones((1,))
        print('u = ', u)
        x = model.traj_to_state(sim_traj)
        #x = model.pred(x, u)
        x = dt_cartpole_dynamics(x, u, dt)
        sim_traj[-1, "u"] = u
        sim_traj = ampc.extend(sim_traj, [x], [[0.0]])

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


memory = Memory("cache")

pendulum = ampc.System(["ang", "angvel"], ["torque"])
cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])

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

def cartpole_dynamics(y, u, g = 1.0, m_c = 1, m_p = 1, L = 1, b = 1.0):
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
    return [omega,
            1.0/(L*(m_c+m_p+m_p*np.sin(theta)**2))*(-u*np.cos(theta) 
                - m_p*L*omega**2*np.cos(theta)*np.sin(theta)
                - (m_c+m_p+m_p)*g*np.sin(theta)
                - b*omega),
            dx,
            1.0/(m_c + m_p*np.sin(theta)**2)*(u + m_p*np.sin(theta)*
                (L*omega**2 + g*np.cos(theta)))]

def dt_cartpole_dynamics(y,u,dt,g=9.8,m=1,L=1,b=1.0):
    y[0] += np.pi
    sol = solve_ivp(lambda t, y: cartpole_dynamics(y, u, g, m, L, b), (0, dt), y, t_eval = [dt])
    if not sol.success:
        raise Exception("Integration failed due to {}".format(sol.message))
    y = sol.y.reshape((4,))
    y[0] -= np.pi
    return sol.y.reshape((4,))

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
        theta0 = rng.uniform(-0.02, 0.02, 1)[0]
        y = [theta0, 0.0, 0.0, 0.0]
        traj = ampc.zeros(cartpole, 400)
        for i in range(400):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 1)
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=['koopman', 'sindy-pendulum', 
        "sindy-cartpole"], help='Specify which system id to test')
    args = parser.parse_args()
    if args.model == 'koopman':
        test_dummy()
    if args.model == 'sindy-pendulum':
        test_sindy_pendulum()
    if args.model == 'sindy-cartpole':
        test_sindy_cartpole()
