from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import array_ops


def vgg16_fcn(inputs, num_classes=2):
    """VGG 16.

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.

    Returns:
        the last op containing the log predictions.
    """
    with tf.variable_scope('vgg_16'):
        net = layers_lib.repeat(inputs, 2, layers.conv2d, 64, [3, 3], scope='conv1')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
        net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
        net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
        net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
        net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
        # Not use fully connected layers.
        net = layers.conv2d(net, 1024, [1, 1], scope='fc6')
        net = layers.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc7')
        net = tf.reduce_mean(net, [1, 2])
    return net
