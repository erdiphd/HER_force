import numpy as np
import random
import sys


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun,**kwargs):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        if replay_strategy == 'final':
            future_t[:] = T

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions, None

    return _sample_her_transitions

def make_sample_her_transitions_contact_energy(replay_strategy, replay_k, reward_fun, **kwargs):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    if kwargs['env_name'] == "FetchPickAndPlace-v1":
        max_limit = np.array([1.55, 1.1, np.inf])
        min_limit = np.array([1.05, 0.4, 0.419])
        contact_energy_function = lambda contact_energy: 100 / (1 + np.exp(-0.1 * contact_energy)) - 50

    elif kwargs['env_name'] == "FetchSlide-v1":
        max_limit = np.array([1.95, 1.2, np.inf])
        min_limit = np.array([0.7, 0.3, 0.4])
        contact_energy_function = lambda contact_energy: 100 / (1 + np.exp(-0.05 * contact_energy)) - 50

    elif kwargs['env_name'] == "FetchPush-v1":
        max_limit = np.array([1.55, 1.11, np.inf])
        min_limit = np.array([1.05, 0.4, 0.4])
        contact_energy_function = lambda contact_energy: 100 / (1 + np.exp(-0.1 * contact_energy)) - 50

    elif kwargs['env_name'] == "FrankaPickAndPlace-v1":
        max_limit = np.array([0.65, 0.35, np.inf])
        min_limit = np.array([0.15, -0.35, 0.03])

    elif kwargs['env_name'] == "FrankaPush-v1":
        max_limit = np.array([0.65, 0.35, np.inf])
        min_limit = np.array([0.15, -0.35, 0.0])

    elif kwargs['env_name'] == "FrankaSlide-v1":
        max_limit = np.array([0.70, 0.35, np.inf])
        min_limit = np.array([0.15, -0.35, 0.0])
    else:
        max_limit = None
        min_limit = None


    def _sample_her_transitions_contact_energy_replay(episode_batch, batch_size_in_transitions):
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        #assert np.any(episode_batch['info_intrinsic_sum_force'][:,:,:] == episode_batch['o'][:, 1:, :]) ,"there is something wrong in the replay buffer force side"
        gripper_position = episode_batch['o'][:,1:,:3]
        limit_checker = np.all((gripper_position > min_limit), axis=-1) & np.all((gripper_position < max_limit), axis=-1).astype(int)

        selected_touch_values = np.multiply(episode_batch['info_intrinsic_sum_energy'], np.expand_dims(limit_checker, axis=-1))


        prev_object_position = episode_batch['o'][:, :-1, 3:6]
        object_position = episode_batch['o'][:, 1:, 3:6]
        object_pos_difference = np.linalg.norm(object_position - prev_object_position,axis=-1)

        sum_through_1_dimension = np.sum(selected_touch_values, axis=-1)
       
        contact_energy = np.multiply(sum_through_1_dimension, object_pos_difference)
       
        contact_energy = contact_energy_function(contact_energy)

        replay_buffer_pri_abs = np.abs(contact_energy)
        replay_buffer_obs_episode = np.sum(replay_buffer_pri_abs, axis=1) + sys.float_info.epsilon
        replay_obs_probability_episode = replay_buffer_obs_episode / np.sum(replay_buffer_obs_episode)
        episode_idxs = np.random.choice(rollout_batch_size, size=batch_size, p=replay_obs_probability_episode)

        if replay_obs_probability_episode.shape[0] < batch_size:
            bias_correction = np.ones(batch_size)
        else:
            probs = replay_obs_probability_episode[episode_idxs]
            tmp = 1 / replay_obs_probability_episode[list(set(episode_idxs))].sum()
            weights = (T * rollout_batch_size * tmp * probs) ** -1
            weights = weights / weights.max()
            bias_correction = weights

        ## TODO check timesteps

        t_samples = np.random.randint(T, size=batch_size)

        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        if replay_strategy == 'final':
            future_t[:] = T

        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions, bias_correction

    return _sample_her_transitions_contact_energy_replay

def make_sample_her_transitions_force(replay_strategy, replay_k, reward_fun, **kwargs):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    if kwargs['env_name'] == "FetchPickAndPlace-v1":
        max_limit = np.array([1.55, 1.1, np.inf])
        min_limit = np.array([1.05, 0.4, 0.419])

    elif kwargs['env_name'] == "FetchSlide-v1":
        max_limit = np.array([1.95, 1.2, np.inf])
        min_limit = np.array([0.7, 0.3, 0.4])

    elif kwargs['env_name'] == "FetchPush-v1":
        max_limit = np.array([1.55, 1.11, np.inf])
        min_limit = np.array([1.05, 0.4, 0.4])

    elif kwargs['env_name'] == "FrankaPickAndPlace-v1":
        max_limit = np.array([0.65, 0.35, np.inf])
        min_limit = np.array([0.15, -0.35, 0.03])

    elif kwargs['env_name'] == "FrankaPush-v1":
        max_limit = np.array([0.65, 0.35, np.inf])
        min_limit = np.array([0.15, -0.35, 0.0])

    elif kwargs['env_name'] == "FrankaSlide-v1":
        max_limit = np.array([0.70, 0.35, np.inf])
        min_limit = np.array([0.15, -0.35, 0.0])

    else:
        max_limit = None
        min_limit = None

    def _sample_her_transitions_force_replay(episode_batch, batch_size_in_transitions):
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        #assert np.any(episode_batch['info_intrinsic_sum_force'][:,:,:] == episode_batch['o'][:, 1:, :]) ,"there is something wrong in the replay buffer force side"

        gripper_position = episode_batch['o'][:,1:,:3]
        limit_checker = np.all((gripper_position > min_limit), axis=-1) & np.all((gripper_position < max_limit), axis=-1).astype(int)

        selected_force_values = np.multiply(episode_batch['info_force_replay_buffer'], np.expand_dims(limit_checker, axis=-1))

        sum_through_3_dimensions = np.sum(selected_force_values,axis=-1)

        ## Work_replay_buffer below 2 lines
        work_done_by_gripper_3d = np.multiply(selected_force_values, gripper_position)
        work_done_by_gripper = np.sum(work_done_by_gripper_3d, axis=-1)


        replay_buffer_pri_abs = np.abs(sum_through_3_dimensions)
        replay_buffer_obs_episode = np.sum(replay_buffer_pri_abs, axis=1) + sys.float_info.epsilon
        replay_obs_probability_episode = replay_buffer_obs_episode / np.sum(replay_buffer_obs_episode)
        episode_idxs = np.random.choice(rollout_batch_size, size=batch_size, p=replay_obs_probability_episode)

        if replay_obs_probability_episode.shape[0] < batch_size:
            bias_correction = np.ones(batch_size)
        else:
            probs = replay_obs_probability_episode[episode_idxs]
            tmp = 1 / replay_obs_probability_episode[list(set(episode_idxs))].sum()
            weights = (T * rollout_batch_size * tmp * probs) ** -0.5
            weights = weights / weights.max()
            bias_correction = weights


        ## TODO check timesteps

        t_samples = np.random.randint(T, size=batch_size)

        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        if replay_strategy == 'final':
            future_t[:] = T

        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions, bias_correction

    return _sample_her_transitions_force_replay

def make_sample_her_transitions_work(replay_strategy, replay_k, reward_fun, **kwargs):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    if kwargs['env_name'] == "FetchPickAndPlace-v1":
        max_limit = np.array([1.55, 1.1, np.inf])
        min_limit = np.array([1.05, 0.4, 0.419])

    elif kwargs['env_name'] == "FetchSlide-v1":
        max_limit = np.array([1.95, 1.2, np.inf])
        min_limit = np.array([0.7, 0.3, 0.4])

    elif kwargs['env_name'] == "FetchPush-v1":
        max_limit = np.array([1.55, 1.11, np.inf])
        min_limit = np.array([1.05, 0.4, 0.4])

    elif kwargs['env_name'] == "FrankaPickAndPlace-v1":
        max_limit = np.array([0.65, 0.35, np.inf])
        min_limit = np.array([0.15, -0.35, 0.03])

    elif kwargs['env_name'] == "FrankaPush-v1":
        max_limit = np.array([0.65, 0.35, np.inf])
        min_limit = np.array([0.15, -0.35, 0.0])

    elif kwargs['env_name'] == "FrankaSlide-v1":
        max_limit = np.array([0.70, 0.35, np.inf])
        min_limit = np.array([0.15, -0.35, 0.0])

    else:
        max_limit = None
        min_limit = None
    def _sample_her_transitions_work_replay(episode_batch, batch_size_in_transitions):
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        #assert np.any(episode_batch['info_intrinsic_sum_force'][:,:,:] == episode_batch['o'][:, 1:, :]) ,"there is something wrong in the replay buffer force side"

        gripper_position = episode_batch['o'][:,1:,:3]
        limit_checker = np.all((gripper_position > min_limit), axis=-1) & np.all((gripper_position < max_limit), axis=-1).astype(int)

        selected_force_values = np.multiply(episode_batch['info_sum_force_replay_buffer'], np.expand_dims(limit_checker, axis=-1))

        sum_through_3_dimensions = np.sum(selected_force_values,axis=-1)

        ## Work_replay_buffer below 2 lines
        work_done_by_gripper_3d = np.multiply(selected_force_values, gripper_position)
        work_done_by_gripper = np.sum(work_done_by_gripper_3d, axis=-1)


        replay_buffer_pri_abs = np.abs(work_done_by_gripper)
        replay_buffer_obs_episode = np.sum(replay_buffer_pri_abs, axis=1) + sys.float_info.epsilon
        replay_obs_probability_episode = replay_buffer_obs_episode / np.sum(replay_buffer_obs_episode)
        episode_idxs = np.random.choice(rollout_batch_size, size=batch_size, p=replay_obs_probability_episode)

        if replay_obs_probability_episode.shape[0] < batch_size:
            bias_correction = np.ones(batch_size)
        else:
            probs = replay_obs_probability_episode[episode_idxs]
            tmp = 1 / replay_obs_probability_episode[list(set(episode_idxs))].sum()
            weights = (T * rollout_batch_size * tmp * probs) ** -0.7
            weights = weights / weights.max()
            bias_correction = weights

        ## TODO check timesteps

        t_samples = np.random.randint(T, size=batch_size)

        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        if replay_strategy == 'final':
            future_t[:] = T

        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions, bias_correction

    return _sample_her_transitions_work_replay


def make_sample_her_transitions_cper(replay_strategy, replay_k, reward_fun,**kwargs):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions_cper(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Up sample transitions with achieved minimal force. 5x higher probability
        force_obs = episode_batch['o'][:, :, -1]  # force_obs = array(buffer_size x T x 1)
        # force_obs_probability: 5 if force_obs > 1, else 1, then normalize
        force_obs_probability = 9 * (force_obs > 1).astype(np.float32) + 1

        # Compute probabilities for episode ...
        force_obs_probability_episode = np.sum(force_obs_probability, axis=1)
        force_obs_probability_episode = force_obs_probability_episode / np.sum(force_obs_probability_episode)
        episode_idxs = np.random.choice(rollout_batch_size, size=batch_size, p=force_obs_probability_episode)

        # pick steps conditioned on episode
        force_obs_probability_step = force_obs_probability[episode_idxs, :]
        force_obs_probability_step = np.delete(force_obs_probability_step, -1, axis=1)
        row_sums = np.sum(force_obs_probability_step, axis=1)
        force_obs_probability_step = force_obs_probability_step / row_sums[:, np.newaxis]
        t_samples = np.array([np.random.choice(T, size=1, p=force_obs_probability_step_row) for force_obs_probability_step_row
                              in list(force_obs_probability_step)])
        t_samples = np.array(t_samples).squeeze()

        # Select which episodes and time steps to use.
        # episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        # t_samples = np.random.randint(T, size=batch_size)

        # Select past (!) time indexes
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        past_offset = np.random.uniform(size=batch_size) * t_samples
        past_offset = past_offset.astype(int)
        future_t = (t_samples + 1)[her_indexes]

        transitions = {key: episode_batch[key][episode_idxs, past_offset].copy()
                       for key in episode_batch.keys()}

        # # Select future time indexes proportional with probability future_p. These
        # # will be used for HER replay by substituting in future goals.
        # her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        # future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        # future_offset = future_offset.astype(int)
        # future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.

        if replay_strategy == 'final':
            future_t[:] = T

        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions, None

    return _sample_her_transitions_cper


def make_sample_her_transitions_entropy(replay_strategy, replay_k, reward_fun,**kwargs):
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, rank_method, temperature, update_stats=False):

        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)

        if not update_stats:

            if rank_method == 'none':
                entropy_trajectory = episode_batch['e']
            else:
                entropy_trajectory = episode_batch['p'] + 0.001
            p_trajectory = np.power(entropy_trajectory, 1 / (temperature + 1e-2))
            p_trajectory = p_trajectory / p_trajectory.sum()
            episode_idxs_entropy = np.random.choice(rollout_batch_size, size=batch_size, replace=True,
                                                    p=p_trajectory.flatten())
            episode_idxs = episode_idxs_entropy

        transitions = {}
        for key in episode_batch.keys():
            if not key == 'p' and not key == 's' and not key == 'e':
                transitions[key] = episode_batch[key][episode_idxs, t_samples].copy()

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        if replay_strategy == 'final':
            future_t[:] = T

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]

        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info

        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions


def make_sample_her_transitions_prioritized_replay(replay_strategy, replay_k, reward_fun,**kwargs):
    if (replay_strategy == 'future') or (replay_strategy == 'final'):
        future_p = 1 - (1. / (1 + replay_k))
    else:
        future_p = 0

    def _sample_proportional(self, rollout_batch_size, batch_size, T):
        episode_idxs = []
        t_samples = []
        for _ in range(batch_size):
            self.n_transitions_stored = min(self.n_transitions_stored, self.size_in_transitions)
            mass = random.random() * self._it_sum.sum(0, self.n_transitions_stored - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            assert idx < self.n_transitions_stored
            episode_idx = idx // T
            assert episode_idx < rollout_batch_size
            t_sample = idx % T
            episode_idxs.append(episode_idx)
            t_samples.append(t_sample)

        return (episode_idxs, t_samples)

    def _sample_her_transitions(self, episode_batch, batch_size_in_transitions, beta):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """

        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        if rollout_batch_size < self.current_size:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
            t_samples = np.random.randint(T, size=batch_size)
        else:
            assert beta >= 0
            episode_idxs, t_samples = _sample_proportional(self, rollout_batch_size, batch_size, T)
            episode_idxs = np.array(episode_idxs)
            t_samples = np.array(t_samples)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.n_transitions_stored) ** (-beta)

        for episode_idx, t_sample in zip(episode_idxs, t_samples):
            p_sample = self._it_sum[episode_idx * T + t_sample] / self._it_sum.sum()
            weight = (p_sample * self.n_transitions_stored) ** (-beta)
            weights.append(weight / max_weight)

        weights = np.array(weights)

        transitions = {}
        for key in episode_batch.keys():
            if not key == "td" and not key == "e":
                episode_batch_key = episode_batch[key].copy()
                transitions[key] = episode_batch_key[episode_idxs, t_samples].copy()

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        if replay_strategy == 'final':
            future_t[:] = T

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]

        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info

        transitions['g'][her_indexes] = future_ag

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info

        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert (transitions['u'].shape[0] == batch_size_in_transitions)

        idxs = episode_idxs * T + t_samples

        return (transitions, weights, idxs)

    return _sample_her_transitions


