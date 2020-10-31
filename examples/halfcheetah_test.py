import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Memory
from pdb import set_trace

import autompc as ampc
import mujoco_py
import gym
import os

memory = Memory("cache")
halfcheetah = ampc.System([f"x{i}" for i in range(18)], [f"u{i}" for i in range(6)])
env = gym.make("HalfCheetah-v2")
halfcheetah.dt = env.dt

def load_buffer(system, env_name="HalfCheetah-v2", buffer_dir="../icra_scripts/buffers2", 
        prefix=None):
    if prefix is None:
        prefix = "Robust_" + env_name + "_0_"
    states = np.load(os.path.join(buffer_dir, prefix+"state.npy"))
    actions = np.load(os.path.join(buffer_dir, prefix+"action.npy"))
    next_states = np.load(os.path.join(buffer_dir, prefix+"next_state.npy"))

    def gym_to_obs(gym):
        #return np.concatenate([[0], gym])
        return gym

    episode_start = 0
    trajs = []
    for i in range(states.shape[0]):
        if i == states.shape[0]-1 or (next_states[i] != states[i+1]).any():
            traj = ampc.empty(system, i - episode_start + 1)
            traj.obs[:] = np.apply_along_axis(gym_to_obs, 1, 
                    states[episode_start:i+1])
            traj.ctrls[:] = actions[episode_start:i+1]
            trajs.append(traj)
            episode_start = i+1
    return trajs


def halfcheetah_dynamics(x, u, n_frames=5):
    old_state = env.sim.get_state()
    old_qpos = old_state[1]
    qpos = x[:len(old_qpos)]
    qvel = x[len(old_qpos):]
    new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
            old_state.act, old_state.udd_state)
    env.sim.set_state(new_state)
    #env.sim.forward()
    env.sim.data.ctrl[:] = u
    for _ in range(n_frames):
        env.sim.step()
    new_qpos = env.sim.data.qpos
    new_qvel = env.sim.data.qvel

    return np.concatenate([new_qpos, new_qvel])

def perf_metric(traj):
    cum_reward = 0.0
    for i in range(len(traj)-1):
        reward_ctrl = -0.1 * np.square(traj[i].ctrl).sum()
        reward_run = (traj[i+1, "x0"] - traj[i, "x0"]) / env.dt
        cum_reward += reward_ctrl + reward_run
    return 200 - cum_reward


@memory.cache
def gen_trajs_uniform_random(num_trajs=1000, traj_len=1000, seed=42):
    rng = np.random.default_rng(seed)
    trajs = []
    for i in range(num_trajs):
        env.seed(int(rng.integers(1 << 30)))
        init_obs = env.reset()
        traj = ampc.zeros(halfcheetah, traj_len)
        traj[0].obs[:] = np.concatenate([[0], init_obs])
        for j in range(1, traj_len):
            action = env.action_space.sample()
            traj[j-1].ctrl[:] = action
            #obs, reward, done, info = env.step(action)
            obs = halfcheetah_dynamics(traj[j-1].obs[:], action)
            traj[j].obs[:] = obs
        trajs.append(traj)
    return trajs


def test_mlp_accuracy():
    #trajs = gen_trajs_uniform_random()
    env = gym.make("HalfCheetah-v2")
    print(f"{env.dt=}")
    halfcheetah.dt = env.dt

    from autompc.evaluators import HoldoutEvaluator, FixedSetEvaluator
    from autompc.metrics import RmseKstepMetric
    from autompc.graphs import KstepGrapher, InteractiveEvalGrapher
    from autompc.sysid import MLP, Koopman

    metric = RmseKstepMetric(halfcheetah, k=10)
    grapher = InteractiveEvalGrapher(halfcheetah, logscale=True)
    grapher2 = KstepGrapher(halfcheetah, kmax=50, kstep=5, evalstep=10)
    rng = np.random.default_rng(42)
    
    evaluator = FixedSetEvaluator(halfcheetah, trajs[:10], metric, rng,
            training_trajs=trajs[10:200])
    evaluator.add_grapher(grapher2)
    cs = MLP.get_configuration_space(halfcheetah)
    cfg = cs.get_default_configuration()
    cfg["n_train_iters"] = 30
    cfg["n_hidden_layers"] = "2"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128
    eval_score, _, graphs = evaluator(MLP, cfg)
    print("eval_score = {}".format(eval_score))
    fig = plt.figure()
    graph = graphs[0]
    graphs[0](fig)
    plt.show()

from autompc.sysid import MLP
@memory.cache
def train_mlp_inner(trajs):
    #trajs = gen_trajs_uniform_random()
    cs = MLP.get_configuration_space(halfcheetah)
    cfg = cs.get_default_configuration()
    cfg["n_hidden_layers"] = "3"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128 # Com
    cfg["hidden_size_3"] = 128
    model = ampc.make_model(halfcheetah, MLP, cfg, use_cuda=True,
            n_train_iters=50)
    model.train(trajs)
    return model.get_parameters()

def train_mlp(trajs):
    cs = MLP.get_configuration_space(halfcheetah)
    cfg = cs.get_default_configuration()
    cfg["n_hidden_layers"] = "3"
    cfg["hidden_size_1"] = 128
    cfg["hidden_size_2"] = 128 # Com
    cfg["hidden_size_3"] = 128
    model = ampc.make_model(halfcheetah, MLP, cfg, use_cuda=True)
    params = train_mlp_inner(trajs)
    model.set_parameters(params)
    #model.net = model.net.to("cpu")
    #model._device = "cpu"
    return model

def test_mlp_ilqr_control():
    env = gym.make("HalfCheetah-v2")
    print(f"{env.dt=}")
    halfcheetah.dt = env.dt
    target_vel = 2.0
    trajs = load_buffer(halfcheetah)
    trajs_rand = gen_trajs_uniform_random(250)
    #model = train_mlp(trajs[:500] + trajs_rand)
    model = train_mlp(trajs[:500])

    # Define task
    from autompc.tasks.task import Task
    from autompc.tasks.quad_cost import QuadCost
    Q = np.zeros((halfcheetah.obs_dim, halfcheetah.obs_dim))
    R = np.eye(halfcheetah.ctrl_dim)
    F = np.zeros((halfcheetah.obs_dim, halfcheetah.obs_dim))
    Q[1,1] = 20.0
    Q[6,6] = 5.0
    Q[7,7] = 5.0
    Q[8,8] = 5.0
    Q[9,9] = 1.0
    R = 0.1 * R
    F[9,9] = 10.0
    x0 = np.zeros(halfcheetah.obs_dim)
    x0[9] = target_vel
    cost = QuadCost(halfcheetah, Q, R, x0=x0)
    #from autompc.tasks.half_cheetah_transformer import HalfCheetahTransformer
    #trans_cfg = {
    #          "target_velocity" : 0.46875,
    #          "u0_log10Rgain" : -1.03125,
    #          "u1_log10Rgain" : 1.15625 ,
    #          "u2_log10Rgain" : 0.71875 ,
    #          "u3_log10Rgain" : 1.59375 ,
    #          "u4_log10Rgain" : -1.90625,
    #          "u5_log10Rgain" : -2.34375,
    #          "x1_log10Fgain" : 1.15625 ,
    #          "x1_log10Qgain" : 3.34375 ,
    #          "x6_log10Qgain" : 0.71875 ,
    #          "x7_log10Qgain" : -1.03125,
    #          "x8_log10Qgain" : 2.90625 ,
    #          "x9_log10Fgain" : 2.46875 ,
    #          "x9_log10Qgain" : 1.15625 
    #        }
    #trans = HalfCheetahTransformer(halfcheetah, **trans_cfg)
    task = Task(halfcheetah)
    task.set_cost(cost)
    task.set_ctrl_bounds(env.action_space.low, env.action_space.high)


    from autompc.control import IterativeLQR
    con = IterativeLQR(halfcheetah, task, model, horizon=20,
            reuse_feedback=5, verbose=True)

    states = []
    sim_traj = ampc.zeros(halfcheetah, 201)
    x = np.concatenate([env.init_qpos, env.init_qvel])
    sim_traj[0].obs[:] = x
    
    constate = con.traj_to_state(sim_traj[:1])
    for step in range(200):
        adjx = x[:]
        u, constate = con.run(constate, x)
        u = np.clip(u, env.action_space.low, env.action_space.high)
        #u = np.zeros(halfcheetah.ctrl_dim)
        #x, reward, _, _ = env.step(u)
        x = halfcheetah_dynamics(x, u)
        print(f"{x=}")
        sim_traj[step].ctrl[:] = u
        sim_traj[step+1].obs[:] = x

    print(f"{perf_metric(sim_traj)=}")
    set_trace()
    #while True:
    #    for i in range(len(sim_traj)):
    #        qpos = sim_traj[i].obs[:9]
    #        qvel = sim_traj[i].obs[9:]
    #        env.set_state(qpos, qvel)
    #        env.render()
    #    time.sleep(1)
    while True:
        env.reset()
        qpos = sim_traj[0].obs[:9]
        qvel = sim_traj[0].obs[9:]
        env.set_state(qpos, qvel)
        for i in range(len(sim_traj)):
            u = sim_traj[i].ctrl
            env.step(u)
            env.render()
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, choices=["mlp_accuracy", "ilqr"],
            default="mlp_accuracy", help="Specify which test to run")
    args = parser.parse_args()
    if args.test == "mlp_accuracy":
        test_mlp_accuracy()
    elif args.test == "ilqr":
        test_mlp_ilqr_control()
    else:
        raise
