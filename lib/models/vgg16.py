from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib

def vgg16_base(inputs):
    with tf.variable_scope('vgg_16'):
        net = layers_lib.repeat(inputs, 2, layers.conv2d, 64, [3, 3], padding='SAME', scope='conv1')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool1')
        net = layers_lib.repeat(net, 2, layers.conv2d, 128, [3, 3], padding='SAME', scope='conv2')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool2')
        net = layers_lib.repeat(net, 3, layers.conv2d, 256, [3, 3], padding='SAME', scope='conv3')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool3')
        net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], padding='SAME', scope='conv4')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool4')
        net = layers_lib.repeat(net, 3, layers.conv2d, 512, [3, 3], padding='SAME', scope='conv5')
        net = layers_lib.max_pool2d(net, [2, 2], scope='pool5')
    return net
