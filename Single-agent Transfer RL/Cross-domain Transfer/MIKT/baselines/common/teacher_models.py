import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow.contrib.layers as layers

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


@register("mlp")
def mlp(freeze=False, num_layers=2, num_hidden=64, activation=tf.tanh,
        layer_norm=False):
    """
    Stack of fully-connected layers to be used in a policy / q-function
    approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """

    def network_fn(X, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            h = tf.layers.flatten(X)
            zs = []
            for i in range(num_layers):
                z = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden,
                       init_scale=np.sqrt(2))
                if layer_norm:
                    z = tf.contrib.layers.layer_norm(z, center=True, scale=True)
                zs.append(z)
                h = activation(z)
            return h, zs

    return network_fn

@register("cnn")
def cnn(freeze=False, **conv_kwargs):
    def network_fn(X, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            scaled_images = tf.cast(X, tf.float32) / 255.
            zs = []
            activ = tf.nn.relu
            z = conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                     **conv_kwargs)
            z = tf.stop_gradient(z) if freeze else z
            zs.append(z)
            h = activ(z)
            z2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs)
            z2 = tf.stop_gradient(z2) if freeze else z2
            zs.append(z2)
            h2 = activ(z2)
            z3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs)
            z3 = tf.stop_gradient(z3) if freeze else z3
            zs.append(z3)
            h3 = activ(z3)
            h3 = conv_to_fc(h3)
            z4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            z4 = tf.stop_gradient(z4) if freeze else z4
            zs.append(z4)
            # return output and hidden layer outputs
            return activ(z4), zs
    return network_fn

@register("cnn_small")
def cnn_small(freeze=False, **conv_kwargs):
    def network_fn(X, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            h = tf.cast(X, tf.float32) / 255.
            activ = tf.nn.relu
            zs = []
            z = conv(h, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs)
            z = tf.stop_gradient(z) if freeze else z
            zs.append(z)
            h = activ(z)
            z = conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs)
            z = tf.stop_gradient(z) if freeze else z
            zs.append(z)
            h = conv_to_fc(activ(z))
            z = fc(h, 'fc1', nh=128, init_scale=np.sqrt(2))
            z = tf.stop_gradient(z) if freeze else z
            zs.append(z)
            h = activ(z)
            # return output and hidden layer outputs
            return h, zs
    return network_fn

@register("conv_only")
def conv_only(freeze=False, convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    ''' 
    convolutions-only net

    Parameters:
    ----------

    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer. 

    Returns:

    function that takes tensorflow tensor as input and returns the output of the last convolutional layer
    
    '''

    def network_fn(X, scope):
        out = tf.cast(X, tf.float32) / 255.
        zs = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for num_outputs, kernel_size, stride in convs:
                z = layers.convolution2d(out,
                                         num_outputs=num_outputs,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         activation_fn=None,
                                         **conv_kwargs)
                z = tf.stop_gradient(z) if freeze else z
                zs.append(z)
                out = tf.nn.relu(z)
            # return output and hidden layer outputs
            return out, zs
    return network_fn

def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

def get_teacher_network_builder(name):
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
