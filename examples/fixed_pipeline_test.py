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

cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
dt = 0.01

Q = np.eye(4)
R = np.eye(1)

cost = QuadCost(cartpole, Q, R)

from autompc.tasks.task import Task

task = Task(cartpole)
task.set_cost(cost)

from autompc.tasks.quad_cost_transformer import QuadCostTransformer

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

from autompc.control import FiniteHorizonLQR
from autompc.sysid import Koopman
from autompc.pipelines import FixedControlPipeline

print(Koopman.get_configuration_space(cartpole))
"""
 Configuration space object:
   Hyperparameters:
     method, Type: Categorical, Choices: {lstsq}, Default: lstsq
     poly_basis, Type: Categorical, Choices: {true, false}, Default: false
     poly_degree, Type: UniformInteger, Range: [2, 8], Default: 3
     product_terms, Type: Categorical, Choices: {false}, Default: false
     trig_basis, Type: Categorical, Choices: {true, false}, Default: false
     trig_freq, Type: UniformInteger, Range: [1, 8], Default: 1
   Conditions:
     poly_degree | poly_basis in {'true'}
     trig_freq | trig_basis in {'true'}
"""
print(QuadCostTransformer.get_configuration_space(cartpole))
"""
  Configuration space object:
    Hyperparameters:
      dx_log10Fgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      dx_log10Qgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      omega_log10Fgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      omega_log10Qgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      theta_log10Fgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      theta_log10Qgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      u_log10Rgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      x_log10Fgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      x_log10Qgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
"""
print(FiniteHorizonLQR.get_configuration_space(cartpole, task, Koopman))
"""
  Configuration space object:
    Hyperparameters:
      horizon, Type: UniformInteger, Range: [1, 1000], Default: 10
"""

pipeline = FixedControlPipeline(cartpole, task, Koopman, FiniteHorizonLQR, 
        [QuadCostTransformer])
cs = pipeline.get_configuration_space()
print(cs)
"""
  Configuration space object:
    Hyperparameters:
      horizon, Type: UniformInteger, Range: [1, 1000], Default: 10
  
  Configuration space object:
    Hyperparameters:
      _controller:horizon, Type: UniformInteger, Range: [1, 1000], Default: 10
      _model:method, Type: Categorical, Choices: {lstsq}, Default: lstsq
      _model:poly_basis, Type: Categorical, Choices: {true, false}, Default: false
      _model:poly_degree, Type: UniformInteger, Range: [2, 8], Default: 3
      _model:product_terms, Type: Categorical, Choices: {false}, Default: false
      _model:trig_basis, Type: Categorical, Choices: {true, false}, Default: false
      _model:trig_freq, Type: UniformInteger, Range: [1, 8], Default: 1
      _task_transformer_0:dx_log10Fgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      _task_transformer_0:dx_log10Qgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      _task_transformer_0:omega_log10Fgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      _task_transformer_0:omega_log10Qgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      _task_transformer_0:theta_log10Fgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      _task_transformer_0:theta_log10Qgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      _task_transformer_0:u_log10Rgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      _task_transformer_0:x_log10Fgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
      _task_transformer_0:x_log10Qgain, Type: UniformFloat, Range: [-3.0, 4.0], Default: 0.0
    Conditions:
      _model:poly_degree | _model:poly_basis in {'true'}
      _model:trig_freq | _model:trig_basis in {'true'}
"""

cfg = cs.get_default_configuration()
controller, model = pipeline(cfg, trajs)

def eval_config(cfg, ret_extra=False):
    print("New configuration===")
    print(cfg)
    # Apply task transformer
    con, model = pipeline(cfg, trajs)

    if np.isnan(con.K).any():
        print("Controller had nans, exiting.")
        return -1.0, 0.0, 0.0
    def run_sim(theta0):
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
            if abs(x[2]) > 30.0:
                print("Simulation out of range.")
                break
        return sim_traj

    theta0_lower = 0.0
    theta0_upper = 3.0
    while theta0_upper - theta0_lower > 0.01:
        theta0 = (theta0_upper + theta0_lower) / 2
        print(f"{theta0=}")
        try:
            sim_traj = func_timeout(5.0, run_sim, args=(theta0,))
            success = (abs(sim_traj[-1, "theta"]) < 0.01
                    and abs(sim_traj[-1, "x"])  < 0.01)
        except FunctionTimedOut:
            print("run_sim timed out")
            success = False
        if success:
            theta0_lower = theta0
        else:
            theta0_upper = theta0
    print("Done with evaluating config!")
    if ret_extra:
        return theta0_lower, run_sim(0.01), con.task
    else:
        return theta0_lower


from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
@memory.cache
def run_smac(seed, runcount_limit=5, n_jobs = 1):
    rng = np.random.RandomState(seed)
    scenario = Scenario({"run_obj": "quality",  
                         "runcount-limit": runcount_limit,  
                         "cs": cs,  
                         "deterministic": "true",
                         "n_jobs" : n_jobs
                         })

    smac = SMAC4HPO(scenario=scenario, rng=rng,
            tae_runner=lambda cfg: -eval_config(cfg))
    
    incumbent = smac.optimize()

    ret_value = dict()
    ret_value["incumbent"] = incumbent
    inc_cost = float("inf")
    inc_costs = []
    costs_and_config_ids = []
    for key, val in smac.runhistory.data.items():
        if val.cost < inc_cost:
            inc_cost = val.cost
        inc_costs.append(inc_cost)
        costs_and_config_ids.append((val.cost, key.config_id))
    ret_value["inc_costs"] = inc_costs
    costs_and_config_ids.sort()
    top_five = [(smac.runhistory.ids_config[cfg_id], cost) for cost, cfg_id 
        in costs_and_config_ids[:5]]
    ret_value["top_five"] = top_five

    return ret_value

ret_value = run_smac(42, runcount_limit=100, n_jobs=1)

#for cfg, cost in ret_value["top_five"]:
#    print(f"Config cost is {cost}")
#    print(cfg)
#
#    theta_max, sim_traj, newtask = eval_config(cfg, ret_extra=True)
#
#    Q, R, F = newtask.get_cost().get_cost_matrices()
#    print("Q")
#    print(Q)
#    print("R")
#    print(R)
#
inc_cfg = ret_value["incumbent"]
#inc_cfg = cs.get_default_configuration()
#
theta_max, sim_traj, newtask = eval_config(cs.get_default_configuration(), ret_extra=True)
print("Incumbent configuration has theta_max={}".format(theta_max))
print("Incumbent configuration is")
print(inc_cfg)

Q, R, F = newtask.get_cost().get_cost_matrices()
print("Q")
print(Q)
print("R")
print(R)

fig = plt.figure()
ax = fig.gca()
ax.set_aspect("equal")
ax.set_xlim([-4.1, 4.1])
ax.set_ylim([-1.1, 1.1])
#set_trace()
ani = animate_cartpole(fig, ax, dt, sim_traj)
ani.save("out/cartpole_test/sep30_03.mp4")

fig = plt.figure()
ax = fig.gca()
ax.plot(range(len(ret_value["inc_costs"])), ret_value["inc_costs"])
ax.set_title("Incumbent cost over time")
ax.set_ylim([-3.0, 0.0])
ax.set_xlabel("Iterations.")
ax.set_ylabel("Cost (-Max Angle)")
plt.show()

#plt.show()



