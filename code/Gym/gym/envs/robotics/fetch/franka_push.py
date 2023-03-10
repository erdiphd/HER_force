import os
from gym import utils
from gym.envs.robotics import franka_fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'frankapush.xml')


class FrankaPushEnv(franka_fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,
            'robot0:torso_lift_joint': -2.24,
            'robot0:head_pan_joint': -0.038,
            'robot0:shoulder_pan_joint': 2.55,
            'robot0:shoulder_lift_joint': -2.68,
            'robot0:upperarm_roll_joint': 0.0,
            'robot0:elbow_flex_joint': 0.984,
            'robot0:forearm_roll_joint': 0.0327,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        franka_fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, force_range=10, force_reward_weight=0.25, task='frankapush', reward_type=reward_type)
        utils.EzPickle.__init__(self)
