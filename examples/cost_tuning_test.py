from pdb import set_trace
import numpy as np
import autompc as ampc
from autompc.tasks.quad_cost import QuadCost
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from joblib import Memory
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
cs = QuadCostTransformer.get_configuration_space(cartpole)
cfg = cs.get_default_configuration()
trans = QuadCostTransformer(cartpole, **cfg.get_dictionary())
newtask = trans(task)

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
model = MyLinear(cartpole)

from autompc.control import FiniteHorizonLQR

def eval_config(cfg):
    # Apply task transformer
    trans = QuadCostTransformer(cartpole, **cfg.get_dictionary())
    newtask = trans(task)

    # Instantiate controller
    cs = FiniteHorizonLQR.get_configuration_space(cartpole, newtask, model)
    cfg = cs.get_default_configuration()
    cfg["horizon"] = 1000
    con = ampc.make_controller(cartpole, newtask, model, FiniteHorizonLQR, cfg)

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
            state_cost = x.T @ Q @ x
            if state_cost > 10000.0:
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
    return theta0_lower, run_sim(theta0_lower), newtask


from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO
@memory.cache
def run_smac(seed, runcount_limit=5, n_jobs = 1):
    rng = np.random.RandomState(seed)
    scenario = Scenario({"run_obj": "quality",  
                         "runcount-limit": runcount_limit,  
                         "cs": cs,  
                         "deterministic": "true",
                         "n_jobs" : n_jobs,
                         "shared_model" : True,
                         })

    smac = SMAC4HPO(scenario=scenario, rng=rng,
            tae_runner=lambda cfg: -eval_config(cfg)[0])
    
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

#ret_value = run_smac(42, runcount_limit=100, n_jobs=10)
#
#for cfg, cost in ret_value["top_five"]:
#    print(f"Config cost is {cost}")
#
#    theta_max, sim_traj, newtask = eval_config(cfg)
#
#    Q, R, F = newtask.get_cost().get_cost_matrices()
#    print("Q")
#    print(Q)
#    print("R")
#    print(R)
#
#inc_cfg = ret_value["incumbent"]
inc_cfg = cs.get_default_configuration()

theta_max, sim_traj, newtask = eval_config(inc_cfg)

Q, R, F = newtask.get_cost().get_cost_matrices()
print("Q")
print(Q)
print("R")
print(R)

#fig = plt.figure()
#ax = fig.gca()
#ax.plot(range(len(ret_value["inc_costs"])), ret_value["inc_costs"])
#ax.set_title("Incumbent cost over time")
#ax.set_ylim([-3.0, 0.0])
#ax.set_xlabel("Iterations.")
#ax.set_ylabel("Cost (-Max Angle)")
#plt.show()

fig = plt.figure()
ax = fig.gca()
ax.set_aspect("equal")
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
#set_trace()
ani = animate_cartpole(fig, ax, dt, sim_traj)
#ani.save("out/cartpole_test/aug18_02.mp4")
plt.show()



