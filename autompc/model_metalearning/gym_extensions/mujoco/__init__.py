import gym
import os

# custom_envs = {
#             # Pusher modifications
#             "PusherMovingGoal-v0":
#                 dict(path='gym_extensions.continuous.mujoco.modified_arm:PusherMovingGoalEnv',
#                      max_episode_steps=100,
#                      reward_threshold=0.0,
#                      kwargs= dict()),
#             # Pusher modifications
#             "PusherLeftSide-v0":
#                 dict(path='gym_extensions.continuous.mujoco.modified_arm:PusherLeftSide',
#                      max_episode_steps=100,
#                      reward_threshold=0.0,
#                      kwargs= dict()),
#             "PusherFullRange-v0":
#                 dict(path='gym_extensions.continuous.mujoco.modified_arm:PusherFullRange',
#                      max_episode_steps=100,
#                      reward_threshold=0.0,
#                      kwargs= dict()),
#             # Striker
#             "StrikerMovingStart-v0":
#                 dict(path='gym_extensions.continuous.mujoco.modified_arm:StrikerMovingStartStateEnv',
#                      max_episode_steps=100,
#                      reward_threshold=0.0,
#                      kwargs= dict()),


#             ### Environment with walls
#             "AntMaze-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_ant:AntMaze',
#                      max_episode_steps=1000,
#                      reward_threshold=3800.0,
#                      kwargs= dict()),
#             "HopperStairs-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_hopper:HopperStairs',
#                      max_episode_steps=1000,
#                      reward_threshold=3800.0,
#                      kwargs= dict()),
#             "HopperSimpleWall-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_hopper:HopperSimpleWallEnv',
#                      max_episode_steps=1000,
#                      reward_threshold=3800.0,
#                      kwargs= dict()),

#             "HopperWithSensor-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_hopper:HopperWithSensorEnv',
#                      max_episode_steps=1000,
#                      reward_threshold=3800.0,
#                      kwargs= dict(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/hopper.xml")),
#             "Walker2dWall-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_walker2d:Walker2dWallEnv',
#                      max_episode_steps=1000,
#                      kwargs= dict()),
#             "Walker2dWithSensor-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_walker2d:Walker2dWithSensorEnv',
#                      max_episode_steps=1000,
#                      kwargs= dict(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/walker2d.xml")),
#             "HalfCheetahWall-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_half_cheetah:HalfCheetahWallEnv',
#                      max_episode_steps=1000,
#                      reward_threshold=4800.0,
#                      kwargs= dict()),
#             "HalfCheetahWithSensor-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_half_cheetah:HalfCheetahWithSensorEnv',
#                      max_episode_steps=1000,
#                      reward_threshold=4800.0,
#                      kwargs= dict(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/half_cheetah.xml")),
#             "HumanoidWall-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_humanoid:HumanoidWallEnv',
#                      max_episode_steps=1000,
#                      kwargs= dict()),
#             "HumanoidWithSensor-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_humanoid:HumanoidWithSensorEnv',
#                      max_episode_steps=1000,
#                      kwargs= dict(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/humanoid.xml")),
#             "HumanoidStandupWithSensor-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_humanoid:HumanoidStandupWithSensorEnv',
#                      max_episode_steps=1000,
#                      kwargs= dict(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/humanoidstandup.xml")),
#             "HumanoidStandupAndRunWall-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_humanoid:HumanoidStandupAndRunWallEnv',
#                      max_episode_steps=1000,
#                      kwargs= dict()),
#             "HumanoidStandupAndRunWithSensor-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_humanoid:HumanoidStandupAndRunEnvWithSensor',
#                      max_episode_steps=1000,
#                      kwargs= dict(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/humanoidstandup.xml")),
#             "HumanoidStandupAndRun-v0" :
#                 dict(path='gym_extensions.continuous.mujoco.modified_humanoid:HumanoidStandupAndRunEnv',
#                      max_episode_steps=1000,
#                      kwargs= dict()),

#                      }

custom_envs = {
        
    # Modified gravity - HalfCheetah
    "HalfCheetahGravityHalf-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahGravityEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(gravity=-4.905)),
    "HalfCheetahGravityThreeQuarters-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahGravityEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(gravity=-7.3575)),
    "HalfCheetahGravityOneAndHalf-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahGravityEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(gravity=-14.715)),
    "HalfCheetahGravityOneAndQuarter-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahGravityEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(gravity=-12.2625)),
    
    # Modified gravity - Hopper
    "HopperGravityHalf-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperGravityEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0,
            kwargs= dict(gravity=-4.905)),
    "HopperGravityThreeQuarters-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperGravityEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0,
            kwargs= dict(gravity=-7.3575)),
    "HopperGravityOneAndHalf-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperGravityEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0,
            kwargs= dict(gravity=-14.715)),
    "HopperGravityOneAndQuarter-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperGravityEnv',
            max_episode_steps=1000,
            reward_threshold=3800.0,
            kwargs= dict(gravity=-12.2625)),
        
    # Modified gravity - Walker2d
    "Walker2dGravityHalf-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dGravityEnv',
                max_episode_steps=1000,
                kwargs= dict(gravity=-4.905)),
    "Walker2dGravityThreeQuarters-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dGravityEnv',
                max_episode_steps=1000,
                kwargs= dict(gravity=-7.3575)),
    "Walker2dGravityOneAndHalf-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dGravityEnv',
                max_episode_steps=1000,
                kwargs= dict(gravity=-14.715)),
    "Walker2dGravityOneAndQuarter-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dGravityEnv',
                max_episode_steps=1000,
                kwargs= dict(gravity=-12.2625)),
        
    # Modified gravity - Humanoid
    "HumanoidGravityHalf-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidGravityEnv',
                max_episode_steps=1000,
                kwargs= dict(gravity=-4.905)),
    "HumanoidGravityThreeQuarters-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidGravityEnv',
                max_episode_steps=1000,
                kwargs= dict(gravity=-7.3575)),
    "HumanoidGravityOneAndHalf-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidGravityEnv',
                max_episode_steps=1000,
                kwargs= dict(gravity=-14.715)),
    "HumanoidGravityOneAndQuarter-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidGravityEnv',
                max_episode_steps=1000,
                kwargs= dict(gravity=-12.2625)),
    
    #--------------------------------------------------
    # Modified body parts - HalfCheetah
    "HalfCheetahBigTorso-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(body_parts=["torso"], size_scale=1.25)),
    "HalfCheetahBigThigh-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(body_parts=["fthigh", "bthigh"], size_scale=1.25)),
    "HalfCheetahBigLeg-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(body_parts=["fshin", "bshin"], size_scale=1.25)),
    "HalfCheetahBigFoot-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(body_parts=["ffoot", "bfoot"], size_scale=1.25)),
    "HalfCheetahSmallTorso-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(body_parts=["torso"], size_scale=.75)),
    "HalfCheetahSmallThigh-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(body_parts=["fthigh", "bthigh"], size_scale=.75)),
    "HalfCheetahSmallLeg-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(body_parts=["fshin", "bshin"], size_scale=.75)),
    "HalfCheetahSmallFoot-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(body_parts=["ffoot", "bfoot"], size_scale=.75)),
    "HalfCheetahSmallHead-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(body_parts=["head"], size_scale=.75)),
    "HalfCheetahBigHead-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=4800.0,
                kwargs= dict(body_parts=["head"], size_scale=1.25)),
    
    # Modified body parts - Hopper
    "HopperBigTorso-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=3800.0,
                kwargs= dict(body_parts=["torso_geom"], size_scale=1.25)),
    "HopperBigThigh-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=3800.0,
                kwargs= dict(body_parts=["thigh_geom"], size_scale=1.25)),
    "HopperBigLeg-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=3800.0,
                kwargs= dict(body_parts=["leg_geom"], size_scale=1.25)),
    "HopperBigFoot-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=3800.0,
                kwargs= dict(body_parts=["foot_geom"], size_scale=1.25)),
    "HopperSmallTorso-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=3800.0,
                kwargs= dict(body_parts=["torso_geom"], size_scale=.75)),
    "HopperSmallThigh-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=3800.0,
                kwargs= dict(body_parts=["thigh_geom"], size_scale=.75)),
    "HopperSmallLeg-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=3800.0,
                kwargs= dict(body_parts=["leg_geom"], size_scale=.75)),
    "HopperSmallFoot-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_hopper:HopperModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                reward_threshold=3800.0,
                kwargs= dict(body_parts=["foot_geom"], size_scale=.75)),
    
    # Modified body parts - Walker2d
    "Walker2dBigTorso-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["torso_geom"], size_scale=1.25)),
    "Walker2dBigThigh-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["thigh_geom", "thigh_left_geom"], size_scale=1.25)),
    "Walker2dBigLeg-v2" :
        dict(path='gautompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["leg_geom", "leg_left_geom"], size_scale=1.25)),
    "Walker2dBigFoot-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["foot_geom", "foot_left_geom"], size_scale=1.25)),
    "Walker2dSmallTorso-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["torso_geom"], size_scale=.75)),
    "Walker2dSmallThigh-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["thigh_geom", "thigh_left_geom"], size_scale=.75)),
    "Walker2dSmallLeg-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["leg_geom", "leg_left_geom"], size_scale=.75)),
    "Walker2dSmallFoot-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_walker2d:Walker2dModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["foot_geom", "foot_left_geom"], size_scale=.75)),
    
    # Modified body parts - Humanoid
    "HumanoidBigTorso-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["torso1", "uwaist", "lwaist"], size_scale=1.25)),
    "HumanoidBigThigh-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["right_thigh1", "left_thigh1", "butt"], size_scale=1.25)),
    "HumanoidBigLeg-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["right_shin1", "left_shin1"], size_scale=1.25)),
    "HumanoidBigFoot-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["left_foot", "right_foot"], size_scale=1.25)),
    "HumanoidSmallTorso-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["torso1", "uwaist", "lwaist"], size_scale=.75)),
    "HumanoidSmallThigh-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["right_thigh1", "left_thigh1", "butt"], size_scale=.75)),
    "HumanoidSmallLeg-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["right_shin1", "left_shin1"], size_scale=.75)),
    "HumanoidSmallFoot-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["left_foot", "right_foot"], size_scale=.75)),
    "HumanoidSmallHead-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["head"], size_scale=.75)),
    "HumanoidBigHead-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["head"], size_scale=1.25)),
    "HumanoidSmallArm-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["right_uarm1", "right_larm", "left_uarm1", "left_larm"], size_scale=.75)),
    "HumanoidBigArm-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["right_uarm1", "right_larm", "left_uarm1", "left_larm"], size_scale=1.25)),
    "HumanoidSmallHand-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["left_hand", "right_hand"], size_scale=.75)),
    "HumanoidBigHand-v2" :
        dict(path='autompc.model_metalearning.gym_extensions.mujoco.modified_humanoid:HumanoidModifiedBodyPartSizeEnv',
                max_episode_steps=1000,
                kwargs= dict(body_parts=["left_hand", "right_hand"], size_scale=1.25)),
}


def register_custom_envs():
    for key, value in custom_envs.items():
        arg_dict = dict(id=key,
                        entry_point=value["path"],
                        max_episode_steps=value["max_episode_steps"],
                        kwargs=value["kwargs"])

        if "reward_threshold" in value:
            arg_dict["reward_threshold"] = value["reward_threshold"]

        gym.envs.register(**arg_dict)

register_custom_envs()
