import numpy as np
import pickle
import os

from autompc.benchmarks.meta_benchmarks.gym_mujoco import GymBenchmark
from autompc.benchmarks.meta_benchmarks.metaworld import MetaBenchmark

gym_names = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Swimmer-v2", "InvertedPendulum-v2", 
              "Reacher-v2", "InvertedDoublePendulum-v2", 
              "Ant-v2", "Humanoid-v2", "HumanoidStandup-v2"]

metaworld_names = ['assembly-v2', 'basketball-v2', 'bin-picking-v2', 'box-close-v2', 
                'button-press-topdown-v2', 'button-press-topdown-wall-v2', 'button-press-v2', 
                'button-press-wall-v2', 'coffee-button-v2', 'coffee-pull-v2', 'coffee-push-v2', 
                'dial-turn-v2', 'disassemble-v2', 'door-close-v2', 'door-lock-v2', 'door-open-v2', 
                'door-unlock-v2', 'hand-insert-v2', 'drawer-close-v2', 'drawer-open-v2', 'faucet-open-v2', 
                'faucet-close-v2', 'hammer-v2', 'handle-press-side-v2', 'handle-press-v2', 'handle-pull-side-v2', 
                'handle-pull-v2', 'lever-pull-v2', 'peg-insert-side-v2', 'pick-place-wall-v2', 'pick-out-of-hole-v2', 
                'reach-v2', 'push-back-v2', 'push-v2', 'pick-place-v2', 'plate-slide-v2', 'plate-slide-side-v2', 
                'plate-slide-back-v2', 'plate-slide-back-side-v2', 'peg-unplug-side-v2', 'soccer-v2', 'stick-push-v2', 
                'stick-pull-v2', 'push-wall-v2', 'reach-wall-v2', 'shelf-place-v2', 'sweep-into-v2', 
                'sweep-v2', 'window-open-v2', 'window-close-v2']

def generate_save_data(path, name, seed=100, n_trajs=100, traj_len=200):
    if name in gym_names:
        benchmark = GymBenchmark(name=name)
    elif name in metaworld_names:
        benchmark = MetaBenchmark(name=name)
    else:
        raise NotImplementedError("Not supported data: {}".format(name))

    system = benchmark.system
    trajs = benchmark.gen_trajs(seed=seed, n_trajs=n_trajs, traj_len=traj_len)
    
    # Save data
    data_name = name + '.pkl'
    output_file_name = os.path.join(path, data_name)
    print("Dumping to ", output_file_name)
    data = {'system': system, 'trajs': trajs}
    with open(output_file_name, 'wb') as fh:
        pickle.dump(data, fh)
    
def load_data(path, name):
    data_name = name + '.pkl'
    input_file_name = os.path.join(path, data_name)
    with open(input_file_name, 'rb') as fh:
        data = pickle.load(fh)
        system = data['system']
        trajs = data['trajs']
    return system, trajs