import tensorflow as tf

ROOT_POS_END = 5
FIRST_TWO_POS_START, FIRST_TWO_POS_END = 5, 11
SECOND_TWO_POS_START, SECOND_TWO_POS_END = 11, 17
THIRD_TWO_POS_START, THIRD_TWO_POS_END = 17, 23
FOURTH_TWO_POS_START, FOURTH_TWO_POS_END = 23, 27

ROOT_VEL_START, ROOT_VEL_END = 27, 33
FIRST_TWO_VEL_START, FIRST_TWO_VEL_END = 33, 39
SECOND_TWO_VEL_START, SECOND_TWO_VEL_END = 39, 45
THIRD_TWO_VEL_START, THIRD_TWO_VEL_END = 45, 51
FOURTH_TWO_VEL_START, FOURTH_TWO_VEL_END = 51, 55

FORCES_ROOT_START, FORCES_ROOT_END = 55, 61
FIRST_TWO_FORCES_START, FIRST_TWO_FORCES_END = 61, 91
SECOND_TWO_FORCES_START, SECOND_TWO_FORCES_END = 91, 121
THIRD_TWO_FORCES_START, THIRD_TWO_FORCES_END = 121, 151
FOURTH_TWO_FORCES_START, FOURTH_TWO_FORCES_END = 151, 181


def eight_to_two_fours_v0(obs):
    # weird indexing stuff
    # takes first four legs and last four legs
    root_qpos, root_qvel = obs[:,:ROOT_POS_END], obs[:,ROOT_VEL_START:ROOT_VEL_END]
    first_four_qpos = obs[:,FIRST_TWO_POS_START:SECOND_TWO_POS_END-2]
    last_four_qpos = obs[:,THIRD_TWO_POS_START:FOURTH_TWO_POS_END]

    first_four_qvel = obs[:,FIRST_TWO_VEL_START:SECOND_TWO_VEL_END-2]
    last_four_qvel = obs[:,THIRD_TWO_VEL_START:FOURTH_TWO_VEL_END]

    forces_root = obs[:,FORCES_ROOT_START:FORCES_ROOT_END]
    first_four_forces = obs[:,FIRST_TWO_FORCES_START:SECOND_TWO_FORCES_END]
    last_four_forces = obs[:,THIRD_TWO_FORCES_START:]

    first_obs = tf.concat([root_qpos, first_four_qpos, root_qvel, first_four_qvel,
                           forces_root, first_four_forces], axis=1)
    second_obs = tf.concat([root_qpos, last_four_qpos, root_qvel,
                            last_four_qvel, forces_root, last_four_forces],
                            axis=1)
    return first_obs, second_obs

def eight_to_two_fours_v1(obs):
    # takes first two and third two legs, and second two and fourth two legs
    root_qpos, root_qvel = obs[:,:ROOT_POS_END], obs[:,ROOT_VEL_START:ROOT_VEL_END]
    first_two_qpos = obs[:,FIRST_TWO_POS_START:FIRST_TWO_POS_END]
    third_two_qpos = obs[:,THIRD_TWO_POS_START:THIRD_TWO_POS_END-2]
    second_two_qpos = obs[:,SECOND_TWO_POS_START:SECOND_TWO_POS_END]
    fourth_two_qpos = obs[:,FOURTH_TWO_POS_START:FOURTH_TWO_POS_END]

    first_two_qvel = obs[:,FIRST_TWO_VEL_START:FIRST_TWO_VEL_END]
    third_two_qvel = obs[:,THIRD_TWO_VEL_START:THIRD_TWO_VEL_END-2]
    second_two_qvel = obs[:,SECOND_TWO_VEL_START:SECOND_TWO_VEL_END]
    fourth_two_qvel = obs[:,FOURTH_TWO_VEL_START:FOURTH_TWO_VEL_END]

    forces_root = obs[:,FORCES_ROOT_START:FORCES_ROOT_END]
    first_two_forces = obs[:,FIRST_TWO_FORCES_START:FIRST_TWO_FORCES_END]
    third_two_forces = obs[:,THIRD_TWO_FORCES_START:THIRD_TWO_FORCES_END]
    second_two_forces = obs[:,SECOND_TWO_FORCES_START:SECOND_TWO_FORCES_END]
    fourth_two_forces = obs[:,FOURTH_TWO_FORCES_START:FOURTH_TWO_FORCES_END]

    first_obs = tf.concat([root_qpos, first_two_qpos, third_two_qpos, root_qvel,
                           first_two_qvel, third_two_qvel, forces_root,
                           first_two_forces, third_two_forces], axis=1)
    second_obs = tf.concat([root_qpos, second_two_qpos, fourth_two_qpos, root_qvel,
                            second_two_qvel, fourth_two_qvel, forces_root,
                            second_two_forces, fourth_two_forces], axis=1)
    return first_obs, second_obs

def eight_to_two_fours_v2(obs):
    # takes first two and fourth two legs, and second two and third two legs
    root_qpos, root_qvel = obs[:,:ROOT_POS_END], obs[:,ROOT_VEL_START:ROOT_VEL_END]
    first_two_qpos = obs[:,FIRST_TWO_POS_START:FIRST_TWO_POS_END]
    fourth_two_qpos = obs[:,FOURTH_TWO_POS_START:FOURTH_TWO_POS_END]
    second_two_qpos = obs[:,SECOND_TWO_POS_START:SECOND_TWO_POS_END]
    third_two_qpos = obs[:,THIRD_TWO_POS_START:THIRD_TWO_POS_END-2]

    first_two_qvel = obs[:,FIRST_TWO_VEL_START:FIRST_TWO_VEL_END]
    fourth_two_qvel = obs[:,FOURTH_TWO_VEL_START:FOURTH_TWO_VEL_END]
    second_two_qvel = obs[:,SECOND_TWO_VEL_START:SECOND_TWO_VEL_END]
    third_two_qvel = obs[:,THIRD_TWO_VEL_START:THIRD_TWO_VEL_END-2]

    forces_root = obs[:,FORCES_ROOT_START:FORCES_ROOT_END]
    first_two_forces = obs[:,FIRST_TWO_FORCES_START:FIRST_TWO_FORCES_END]
    fourth_two_forces = obs[:,FOURTH_TWO_FORCES_START:FOURTH_TWO_FORCES_END]
    second_two_forces = obs[:,SECOND_TWO_FORCES_START:SECOND_TWO_FORCES_END]
    third_two_forces = obs[:,THIRD_TWO_FORCES_START:THIRD_TWO_FORCES_END]
    second_two_forces = obs[:,SECOND_TWO_FORCES_START:SECOND_TWO_FORCES_END]

    first_obs = tf.concat([root_qpos, first_two_qpos, fourth_two_qpos, root_qvel,
                           first_Two_qvel, fourth_two_qvel, forces_root,
                           first_two_forces, fourth_two_forces], axis=1)
    second_obs = tf.concat([root_qpos, second_two_qpos, third_two_qpos, root_qvel,
                            second_two_qvel, third_two_qvel, forces_root,
                            second_two_forces, third_two_forces], axis=1)
    return first_obs, second_obs

def eight_to_two_sixes(obs):
    root_qpos, root_qvel = obs[:,:ROOT_POS_END], obs[:,ROOT_VEL_START:ROOT_VEL_END]
    first_six_qpos = obs[:,FIRST_TWO_POS_START:THIRD_TWO_POS_END-2]
    last_six_qpos = obs[:,SECOND_TWO_POS_START:FOURTH_TWO_POS_END]

    first_six_qvel = obs[:,FIRST_TWO_VEL_START:THIRD_TWO_VEL_END-2]
    last_six_qvel = obs[:,SECOND_TWO_VEL_START:FOURTH_TWO_VEL_END]

    forces_root = obs[:,FORCES_ROOT_START:FORCES_ROOT_END]
    first_six_forces = obs[:,FIRST_TWO_FORCES_START:THIRD_TWO_FORCES_END]
    last_six_forces = obs[:,SECOND_TWO_FORCES_START:FOURTH_TWO_FORCES_END]

    first_obs = tf.concat([root_qpos, first_six_qpos, root_qvel, first_six_qvel,
                           forces_root, first_six_forces], axis=1)
    second_obs = tf.concat([root_qpos, last_six_qpos, root_qvel, last_six_qvel,
                            forces_root, last_six_forces], axis=1)
    return first_obs, second_obs


def eight_to_four_and_six(obs):
    root_qpos, root_qvel = obs[:,:ROOT_POS_END], obs[:,ROOT_VEL_START:ROOT_VEL_END]
    first_four_qpos = obs[:,FIRST_TWO_POS_START:SECOND_TWO_POS_END-2]
    last_six_qpos = obs[:,SECOND_TWO_POS_START:FOURTH_TWO_POS_END]

    first_four_qvel = obs[:,FIRST_TWO_VEL_START:SECOND_TWO_VEL_END-2]
    last_six_qvel = obs[:,SECOND_TWO_VEL_START:FOURTH_TWO_VEL_END]

    forces_root = obs[:,FORCES_ROOT_START:FORCES_ROOT_END]
    first_four_forces = obs[:,FIRST_TWO_FORCES_START:SECOND_TWO_FORCES_END]
    last_six_forces = obs[:,SECOND_TWO_FORCES_START:FOURTH_TWO_FORCES_END]

    first_obs = tf.concat([root_qpos, first_four_qpos, root_qvel, first_four_qvel,
                           forces_root, first_four_forces], axis=1)
    second_obs = tf.concat([root_qpos, last_six_qpos, root_qvel, last_six_qvel,
                            forces_root, last_six_forces], axis=1)
    return first_obs, second_obs

def get_fixed_mapping(mapping):
    if mapping == 'eight_to_two_fours_v0':
        return eight_to_two_fours_v0
    elif mapping == 'eight_to_two_fours_v1':
        return eight_to_two_fours_v1
    elif mapping == 'eight_to_two_fours_v2':
        return eight_to_two_fours_v2
    elif mapping == 'eight_to_two_sixes':
        return eight_to_two_sixes
    elif mapping == 'eight_to_four_and_six':
        return eight_to_four_and_six
    else:
        raise NotImplementedError
