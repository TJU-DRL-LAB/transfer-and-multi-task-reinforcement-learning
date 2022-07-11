import numpy as np
import tensorflow as tf
from baselines.a2c.utils import fc

def mlp(source_dim, num_layers=2, num_hidden=64, activation=tf.tanh):
    """Encoder to generate embedding to feed into teacher"""
    def network_fn(X, scope):
        h = tf.layers.flatten(X)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for i in range(num_layers):
                h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden,
                       init_scale=np.sqrt(2))
                h = activation(h)
            embedding = fc(h, 'mlp_out', nh=source_dim, init_scale=np.sqrt(2))
            return embedding
    return network_fn

