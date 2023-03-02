import numpy as np

from gym.envs.robotics import rotations, robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range, task,
        distance_threshold, initial_qpos, force_range, force_reward_weight, reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.task = task
        self.baseline = True
        self.target_range_x = 0.1  # entire table: 0.125
        self.target_range_y = 0.1  # entire table: 0.175

        self.obj_range_x = 0.1
        self.obj_range_y = 0.1
        
        self.force_range = force_range
        self.force_reward_weight = force_reward_weight
        
        if reward_type=='intrinsic':
            self.baseline = False
        else:
            self.baseline = True

        self.n_action = 4
        

        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos, task=task)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute rewards
        d_pos = goal_distance(achieved_goal, goal)
        position_reward = -(d_pos > self.distance_threshold).astype(np.float32)
        if self.reward_type == 'sparse':
            return position_reward
        elif self.reward_type == 'intrinsic':
            # weights
            intrinsic = info['intrinsic_sum_force']
            force_reward = -(np.squeeze(intrinsic) < self.force_range).astype(np.float32)
            force_weight = self.force_reward_weight
            position_weight = 1 - force_weight
            return force_weight * force_reward + position_weight * position_reward
        elif self.reward_type == 'continuous':
            return d_pos
        else:
            raise ValueError('False reward type is selected. Select either intrinsic, continuous or sparse')
            return None

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0_finger_joint1', 0.)
            self.sim.data.set_joint_qpos('robot0_finger_joint2', 0.)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [0, 1., 0., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
        if self.sim.data.mocap_pos[0][-1] < 0:
            self.sim.data.mocap_pos[0][-1] = 0

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        # forces
        force = self.force
        sum_force = self.sum_force

        force_replay_buffer = self.force_replay_buffer
        sum_force_replay_buffer = self.sum_force_replay_buffer

        touch_replay_buffer = float(self.touch_replay_buffer)
        sum_touch_replay_buffer = float(self.sum_touch_replay_buffer)

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())

        if self.baseline:
            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
                object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel])
        else:
            obs = np.concatenate([
                grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
                object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel, force.ravel(), sum_force.ravel()])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'force_replay_buffer': [force_replay_buffer, sum_force_replay_buffer],
            'touch_replay_buffer': [touch_replay_buffer, sum_touch_replay_buffer],
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.init_center[:2]
            object_xpos[0] += self.np_random.uniform(-self.obj_range_x, self.obj_range_x)
            object_xpos[1] += self.np_random.uniform(-self.obj_range_y, self.obj_range_y)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            object_qpos[3] = 1
            object_qpos[4] = 0
            object_qpos[5] = 0
            object_qpos[6] = 0
            object_qpos[2] = 0.025
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            object_qvel = self.sim.data.get_joint_qvel('object0:joint')
            self.sim.data.set_joint_qvel('object0:joint', 0*object_qvel)
        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.target_center.copy()

        goal[1] += self.np_random.uniform(-self.target_range_y, self.target_range_y)
        goal[0] += self.np_random.uniform(-self.target_range_x, self.target_range_x)
        goal[2] = self.height_offset
        if self.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)

        goal += self.target_offset
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        self.target_center = self.sim.data.get_site_xpos('goal_center')
        self.init_center = self.sim.data.get_site_xpos('init_center')

        table_position = self.sim.data.get_body_xpos("table0")
        # Move end effector into position.
        gripper_target = table_position + np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_target = self.init_center + np.array([0, 0, 0.15])
        gripper_rotation = np.array([0., 1., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]



        # It is relative pose with respect to table position
        site_id = self.sim.model.site_name2id('goal_1')
        self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, self.target_range_y,
                                                                 0] - table_position
        site_id = self.sim.model.site_name2id('goal_2')
        self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, self.target_range_y,
                                                                 0] - table_position
        site_id = self.sim.model.site_name2id('goal_3')
        self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, -self.target_range_y,
                                                                 0] - table_position
        site_id = self.sim.model.site_name2id('goal_4')
        self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, -self.target_range_y,
                                                                 0] - table_position

        site_id = self.sim.model.site_name2id('init_1')
        self.sim.model.site_pos[site_id] = self.init_center + [self.obj_range_x, self.obj_range_y, 0.0] - table_position
        site_id = self.sim.model.site_name2id('init_2')
        self.sim.model.site_pos[site_id] = self.init_center + [self.obj_range_x, -self.obj_range_y,
                                                               0.0] - table_position
        site_id = self.sim.model.site_name2id('init_3')
        self.sim.model.site_pos[site_id] = self.init_center + [-self.obj_range_x, self.obj_range_y,
                                                               0.0] - table_position
        site_id = self.sim.model.site_name2id('init_4')
        self.sim.model.site_pos[site_id] = self.init_center + [-self.obj_range_x, -self.obj_range_y,
                                                               0.0] - table_position

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
