import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow.contrib.layers as layers
def lateral_fc(scope, zs, zs1, size):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        ps = [tf.get_variable('p{}'.format(i), (), tf.float32,
                              tf.constant_initializer(0.01))
              for i in range(1, len(zs) + 1)]  # lateral_fc0/p1

        # make student have smaller proportion initially  lateral_fc0/p0
        ps.insert(0, tf.get_variable('p{}'.format(0), (), tf.float32,
                                     tf.constant_initializer(-0.1)))

        weight = [tf.get_variable('w{}'.format(i), (), tf.float32,
                              tf.constant_initializer(0.4))
              for i in range(1, len(zs) + 1)]  # lateral_fc0/p1
        weight.insert(0, tf.get_variable('w{}'.format(0), (), tf.float32,
                                     tf.constant_initializer(0.6)))


        ps = tf.nn.softmax(ps)
        weight = tf.nn.softmax(weight)
        Us = [tf.get_variable('U{}'.format(i), [U.var_shape(z)[1], size],
                              tf.float32)
              for i, z in enumerate(zs)]
        Us1 = [tf.get_variable('U1{}'.format(i), [U.var_shape(z)[1], size],
                              tf.float32)
              for i, z in enumerate(zs1)]


        # reshape so dimensions work out
        teacher_ps = tf.reshape(ps[1:], [len(zs), 1, 1])

        teacher_sum = tf.reduce_sum(tf.matmul(tf.multiply(tf.multiply(teacher_ps, weight[0]), zs), Us),
                                    axis=0)
        teacher_sum += tf.reduce_sum(tf.matmul(tf.multiply(tf.multiply(teacher_ps, weight[1]), zs1), Us1),
                                    axis=0)
        return teacher_sum, ps, weight

""" for all these functions teachers is the actual callable network"""

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk

@register("mlp")
def mlp(encoder, encoder1, encoder_scope, encoder_scope1, teachers, teachers1, teacher_scopes, teacher_scopes1, num_layers=2, num_hidden=64, activation=tf.tanh):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)
    
    num_hidden: int                 size of fully-connected layers (default: 64)
    
    activation:                     activation function (default: tf.tanh)
        
    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X, scope, independent=False):  # scope student0
        h = tf.layers.flatten(X)
        if not independent:
            embedding = encoder(h, encoder_scope)
            embedding1 = encoder1(h, encoder_scope1)
            zs = [t(embedding, s)[1] for t, s in zip(teachers, teacher_scopes)]
            zs1 = [t(embedding1, s)[1] for t, s in zip(teachers1, teacher_scopes1)]
        else:
            embedding = None
            embedding1 = None

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            student_ps = []
            teacher_weight = []
            for i in range(num_layers):
                student_out = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden,
                                 init_scale=np.sqrt(2))
                if independent:
                    h = student_out
                else:
                    teacher_zs = [z[i] for z in zs]
                    teacher_zs1 = [z[i] for z in zs1]

                    teacher_sum, ps, weight = lateral_fc('lateral_fc{}'.format(i),
                                                 teacher_zs, teacher_zs1, num_hidden)
                    student_ps.append(ps[0])
                    teacher_weight.append(weight)

                    h = teacher_sum + ps[0] * student_out
                h = activation(h)
            return h, student_ps, teacher_weight, embedding, embedding1
    return network_fn

def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

def get_student_network_builder(name):
    """
    If you want to register your own network outside models.py, you just need:

    Usage Example:
    -------------
    from baselines.common.models import register
    @register("your_network_name")
    def your_network_define(**net_kwargs):
        ...
        return network_fn

    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))
