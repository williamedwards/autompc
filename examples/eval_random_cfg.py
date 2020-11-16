# Created by William Edwards (wre2@illinois.edu)

# External project includes
import numpy as np
import torch
from joblib import Memory

# Autompc Includes
import autompc as ampc
from autompc.pipelines import FixedControlPipeline
from autompc.control import IterativeLQR
from autompc.tasks import Task, QuadCost, QuadCostTransformer
from autompc.sysid import MLP

memory = Memory("cache")
cartpole = ampc.System(["theta", "omega", "x", "dx"], ["u"])
cartpole.dt = 0.05

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
    y += dt * cartpole_simp_dynamics(y,u[0],g,m,L,b)
    return y

@memory.cache
def gen_trajs(traj_len, num_trajs, dt, seed=42,
        init_min = [-1.0, 0.0, 0.0, 0.0], init_max=[1.0, 0.0, 0.0, 0.0],
        umin=-20.0, umax=20.0):
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(num_trajs):
        state0 = [rng.uniform(minval, maxval, 1)[0] for minval, maxval 
                in zip(init_min, init_max)]
        y = state0[:]
        traj = ampc.zeros(cartpole, traj_len)
        traj.obs[:] = y
        for i in range(traj_len):
            traj[i].obs[:] = y
            u  = rng.uniform(umin, umax, 1)
            y = dt_cartpole_dynamics(y, u, dt)
            traj[i].ctrl[:] = u
        trajs.append(traj)
    return trajs

def perf_metric(traj, threshold=0.2):
    cost = 0.0
    for i in range(len(traj)):
        if (np.abs(traj[i].obs[0]) > threshold 
                or np.abs(traj[i].obs[1]) > threshold):
            cost += 1
    return cost

def runsim(controller, simsteps=200, init_obs=np.array([3.1,0.0,0.0,0.0])):
    sim_traj = ampc.zeros(cartpole, 1)
    x = np.copy(init_obs)
    sim_traj[0].obs[:] = x
    
    constate = controller.traj_to_state(sim_traj)
    for _  in range(simsteps):
        u, constate = controller.run(constate, sim_traj[-1].obs)
        x = dt_cartpole_dynamics(x, u, cartpole.dt)
        print(f"u={u} x={x}")
        sim_traj[-1].ctrl[:] = u
        sim_traj = ampc.extend(sim_traj, [x], 
                np.zeros((1, cartpole.ctrl_dim)))
    return sim_traj

def eval_cfg(pipeline, trajs, cfg):
    torch.manual_seed(42)
    controller, model = pipeline(cfg, trajs)
    traj = runsim(controller)
    return perf_metric(traj)

def get_bad_cfg(cs):
    cfg = cs.get_default_configuration()
    return cfg

def get_good_cfg(cs):
    cfg = cs.get_default_configuration()
    cfg["_controller:horizon"] = 19
    cfg["_model:n_hidden_layers"] = '4'
    cfg["_model:hidden_size_1"] = 68
    cfg["_model:hidden_size_2"] = 55
    cfg["_model:hidden_size_3"] = 49
    cfg["_model:hidden_size_4"] = 32
    cfg["_model:lr_log10"] = -2.0235957107828
    cfg["_model:nonlintype"] = 'selu'
    cfg["_task_transformer_0:dx_log10Fgain"] = 3.5633141666235906
    cfg["_task_transformer_0:dx_log10Qgain"] = -2.469765320813216
    cfg["_task_transformer_0:omega_log10Fgain"] = 1.390670781173414
    cfg["_task_transformer_0:omega_log10Qgain"] = -1.4767322831197298
    cfg["_task_transformer_0:theta_log10Fgain"] = 3.9812219620729516
    cfg["_task_transformer_0:theta_log10Qgain"] = -0.4314357462958984
    cfg["_task_transformer_0:u_log10Rgain"] = 1.6473765298266017
    cfg["_task_transformer_0:x_log10Fgain"] = -2.1126069681670785
    cfg["_task_transformer_0:x_log10Qgain"] = 0.18050003268785453

    return cfg

def main():
    # Initialize task
    Q = np.eye(4)
    R = np.eye(1)
    F = np.eye(4)
    cost = QuadCost(cartpole, Q, R, F)
    task = Task(cartpole)
    task.set_cost(cost)
    task.set_ctrl_bound("u", -20.0, 20.0)


    # Initialize pipeline
    pipeline = FixedControlPipeline(cartpole, task, MLP,
            IterativeLQR, [QuadCostTransformer],
            controller_kwargs={"reuse_feedback" : -1},
            use_cuda=True)

    # Generate trajectories
    init_max = np.array([1.0, 10.0, 1.0, 10.0])
    trajs = gen_trajs(200, 500, dt=cartpole.dt, init_max=init_max,
            init_min=-init_max, seed=42)

    cs = pipeline.get_configuration_space()
    cs.seed(42)
    bad_cfg = get_bad_cfg(cs)
    good_cfg = get_good_cfg(cs)
    rand1_cfg = cs.sample_configuration()
    rand2_cfg = cs.sample_configuration()
    rand3_cfg = cs.sample_configuration()
    bad_cfg_eval = eval_cfg(pipeline, trajs, bad_cfg)
    good_cfg_eval = eval_cfg(pipeline, trajs, good_cfg)
    rand1_cfg_eval = eval_cfg(pipeline, trajs, rand1_cfg)
    rand2_cfg_eval = eval_cfg(pipeline, trajs, rand2_cfg)
    rand3_cfg_eval = eval_cfg(pipeline, trajs, rand3_cfg)
    print("Eval bad_cfg=", bad_cfg_eval)
    print("Eval good_cfg=", good_cfg_eval)
    print("Eval rand1_cfg=", rand1_cfg_eval)
    print("Eval rand2_cfg=", rand2_cfg_eval)
    print("Eval rand3_cfg=", rand3_cfg_eval)

if __name__ == "__main__":
    main()
