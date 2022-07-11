import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.teacher_models import get_teacher_network_builder
from baselines.common.policies import PolicyWithValue

import gym


def build_teacher_policy(env, policy_network, normalize_observations=False,
                         estimate_q=False, freeze=False, **policy_kwargs):
    """
    :type env: environment to run on
    :type policy_network: usually a string specifying the type of network 通常是指定网络类型的字符串
    :type freeze: not actually used, since the teacher parameters aren't included in the
    minimize operation, there's no need to call stop_gradients on the teacher
    """
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_teacher_network_builder(network_type)(freeze, **policy_kwargs)

    def policy_fn(pi_scope, vf_scope=None, nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)
        
        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        policy_latent, zs = policy_network(encoded_x, pi_scope)

        _v_net = policy_network
        vf_latent, vf_zs = _v_net(encoded_x, vf_scope)
        
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
        return policy, zs, vf_zs

    return policy_fn

def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms
    
