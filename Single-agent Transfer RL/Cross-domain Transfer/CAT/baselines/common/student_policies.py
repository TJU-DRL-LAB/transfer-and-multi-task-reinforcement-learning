import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.student_models import get_student_network_builder
from baselines.common.teacher_models import get_teacher_network_builder 
from baselines.common.policies import PolicyWithValue
import baselines.common.encoder as encoder

import gym

def build_student_policy(env, policy_network, teacher_networks, teacher_networks1, mapping,
                         normalize_observations=False, estimate_q=False,
                         source_env=None, source_env1=None, **policy_kwargs):
    """
    :type env: environment to run on
    :type policy_network: usually a string specifying type of network for the
    student
    :type teacher_networks: list of strings specifying the type of networks for
    the teachers
    """
    if isinstance(policy_network, str):
        network_type = policy_network  # mlp

        teachers = [get_teacher_network_builder(t)(True, **policy_kwargs)
                    for t in teacher_networks]
        teachers1 = [get_teacher_network_builder(t)(True, **policy_kwargs)
                    for t in teacher_networks1]
        # print('teachers:', teachers)  # mlp
        # print('teachers1:', teachers1)
        # value func
        teacher_vfs = [get_teacher_network_builder(t)(True, **policy_kwargs)
                       for t in teacher_networks]
        teacher_vfs1 = [get_teacher_network_builder(t)(True, **policy_kwargs)
                       for t in teacher_networks1]

        teacher_scopes = ['teacher{}'.format(i) for i in range(len(teachers))]  # teacher0
        teacher_scopes1 = ['teacher1{}'.format(i) for i in range(len(teachers1))]  # teacher1
        teacher_vf_scopes = ['vf_teacher{}'.format(i) for i in  # vf_teacher0
                             range(len(teacher_vfs))]
        teacher_vf_scopes1 = ['vf_teacher1{}'.format(i) for i in  # vf_teacher1
                             range(len(teacher_vfs1))]
        source_dim = source_env.observation_space.shape[0]
        source_dim1 = source_env1.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        mapper = encoder.mlp(source_dim)  # embedding
        mapper1 = encoder.mlp(source_dim1)  # embedding1
        encoder_scope = 'encoder'
        encoder_scope1 = 'encoder1'

        policy_network = get_student_network_builder(network_type)(
            mapper, mapper1,encoder_scope, encoder_scope1, teachers, teachers1, teacher_scopes, teacher_scopes1, **policy_kwargs)
        vf_network = get_student_network_builder(network_type)(
            mapper, mapper1,encoder_scope, encoder_scope1, teacher_vfs, teacher_vfs1, teacher_vf_scopes, teacher_vf_scopes1,
            **policy_kwargs)
    def policy_fn(pi_scope, vf_scope, nbatch=None, nsteps=None, sess=None,
                  observ_placeholder=None, independent=False):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)
        
        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)
        policy_latent, student_ps, teacher_weight, embedding, embedding1 = policy_network(encoded_x,
                                                                                          pi_scope,
                                                                                          independent)
        vf_latent, vf_student_ps, vf_teacher_weight, _, _ = vf_network(encoded_x,
                                                                       vf_scope,
                                                                       independent)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            pi_scope=pi_scope,
            vf_scope=vf_scope,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy, student_ps, vf_student_ps, teacher_weight, vf_teacher_weight, embedding, embedding1

    return policy_fn


def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms
    
