import numpy as np
import pickle
import os

# from autompc.benchmarks.meta_benchmarks.gym_mujoco import GymBenchmark
# from autompc.benchmarks.meta_benchmarks.metaworld import MetaBenchmark

gym_names = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Swimmer-v2", "InvertedPendulum-v2", 
              "Reacher-v2", "Pusher-v2", "InvertedDoublePendulum-v2", 
              "Ant-v2", "Humanoid-v2", "HumanoidStandup-v2"] #11

gym_small_names = ["HalfCheetahSmall-v2", "HopperSmall-v2", "Walker2dSmall-v2", "SwimmerSmall-v2", "InvertedPendulumSmall-v2", 
                    "ReacherSmall-v2", "InvertedDoublePendulumSmall-v2", 
                    "AntSmall-v2", "HumanoidSmall-v2", "HumanoidStandupSmall-v2"] #11

gym_extensions_names = ["HalfCheetahGravityHalf-v2", "HalfCheetahGravityThreeQuarters-v2", "HalfCheetahGravityOneAndHalf-v2", "HalfCheetahGravityOneAndQuarter-v2",
                        "HopperGravityHalf-v2", "HopperGravityThreeQuarters-v2", "HopperGravityOneAndHalf-v2", "HopperGravityOneAndQuarter-v2",
                        "Walker2dGravityHalf-v2", "Walker2dGravityThreeQuarters-v2", "Walker2dGravityOneAndHalf-v2", "Walker2dGravityOneAndQuarter-v2",
                        "HumanoidGravityHalf-v2", "HumanoidGravityThreeQuarters-v2", "HumanoidGravityOneAndHalf-v2", "HumanoidGravityOneAndQuarter-v2",
                        "HalfCheetahBigTorso-v2", "HalfCheetahBigThigh-v2", "HalfCheetahBigLeg-v2", "HalfCheetahBigFoot-v2", "HalfCheetahBigHead-v2",
                        "HalfCheetahSmallTorso-v2", "HalfCheetahSmallThigh-v2", "HalfCheetahSmallLeg-v2", "HalfCheetahSmallFoot-v2", "HalfCheetahSmallHead-v2", 
                        "HopperBigTorso-v2", "HopperBigThigh-v2", "HopperBigLeg-v2", "HopperBigFoot-v2", 
                        "HopperSmallTorso-v2", "HopperSmallThigh-v2", "HopperSmallLeg-v2", "HopperSmallFoot-v2", 
                        "Walker2dBigTorso-v2", "Walker2dBigThigh-v2", "Walker2dBigLeg-v2", "Walker2dBigFoot-v2",
                        "Walker2dSmallTorso-v2", "Walker2dSmallThigh-v2", "Walker2dSmallLeg-v2", "Walker2dSmallFoot-v2", 
                        "HumanoidBigTorso-v2", "HumanoidBigThigh-v2", "HumanoidBigLeg-v2", "HumanoidBigFoot-v2", "HumanoidBigHead-v2", "HumanoidBigArm-v2", "HumanoidBigHand-v2",
                        "HumanoidSmallTorso-v2", "HumanoidSmallThigh-v2", "HumanoidSmallLeg-v2", "HumanoidSmallFoot-v2", "HumanoidSmallHead-v2", "HumanoidSmallArm-v2", "HumanoidSmallHand-v2",
                        "HalfCheetahWall-v2", "HalfCheetahWithSensor-v2", "HopperSimpleWall-v2", "HopperWithSensor-v2", 
                        "Walker2dWall-v2", "Walker2dWithSensor-v2", "HumanoidWall-v2", "HumanoidWithSensor-v2",
                        "HumanoidStandupWithSensor-v2", "HumanoidStandupAndRunWall-v2", "HumanoidStandupAndRunWithSensor-v2",
                        "HumanoidStandupAndRun-v2", "PusherMovingGoal-v2"] #78

meta_data = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Swimmer-v2", "InvertedPendulum-v2", 
              "Reacher-v2", "Pusher-v2", "InvertedDoublePendulum-v2", "Ant-v2", "Humanoid-v2", 
              "HalfCheetahSmall-v2", "ReacherSmall-v2", "SwimmerSmall-v2",
              "HopperGravityThreeQuarters-v2", "Walker2dGravityOneAndHalf-v2", "HalfCheetahGravityOneAndQuarter-v2",
              "HalfCheetahBigThigh-v2", "HopperSmallLeg-v2", "Walker2dSmallTorso-v2", "PusherMovingGoal-v2"] #20


# def generate_save_data(path, name, seed=100, n_trajs=100, traj_len=200):
#     if name in gym_names:
#         benchmark = GymBenchmark(name=name)
#     elif name in metaworld_names:
#         benchmark = MetaBenchmark(name=name)
#     else:
#         raise NotImplementedError("Not supported data: {}".format(name))

#     system = benchmark.system
#     trajs = benchmark.gen_trajs(seed=seed, n_trajs=n_trajs, traj_len=traj_len)
    
#     # Save data
#     data_name = name + '.pkl'
#     output_file_name = os.path.join(path, data_name)
#     print("Dumping to ", output_file_name)
#     data = {'system': system, 'trajs': trajs}
#     with open(output_file_name, 'wb') as fh:
#         pickle.dump(data, fh)
    
def load_data(path, name):
    data_name = name + '.pkl'
    input_file_name = os.path.join(path, data_name)
    with open(input_file_name, 'rb') as fh:
        data = pickle.load(fh)
        system = data['system']
        trajs = data['trajs']
    return system, trajs