from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib


def vgg16(inputs,
          num_classes=1000,
          is_training=True,
          dropout_keep_prob=0.5):
    """Oxford Net VGG 16-Layers version D Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
        layers during training.

    Returns:
        the last op containing the log predictions and end_points dict.
    """
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
    # Use conv2d instead of fully_connected layers.
    net = layers.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
    net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
    net = layers.conv2d(net, 4096, [1, 1], scope='fc7')
    net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
    net = layers.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
    return net
