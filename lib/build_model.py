from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import array_ops
slim = tf.contrib.slim

from models.vgg16 import vgg16_base


def build_model(model_name, inputs, num_classes=2, is_training=True, dropout_keep_prob=0.5):
    use_fcn = False
    if model_name.find('fcn') >= 0:
        use_fcn = True
        model_base_name = model_name[0:-4]
    else:
        model_base_name = model_name

    if model_base_name == 'vgg16':
        net = vgg16_base(inputs)
    elif model_base_name == 'inception_v1':
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            net, _ = inception.inception_v1_base(inputs)
    elif model_base_name == 'inception_v2':
        with slim.arg_scope(inception.inception_v2_arg_scope()):
            net, _ = inception.inception_v2_base(inputs)
    elif model_base_name == 'inception_v3':
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            net, _ = inception.inception_v3_base(inputs)
    else:
        raise Exception('model {} is not existed'.format(model_name))

    with tf.variable_scope('not_pretrained'):
        if use_fcn:
            net = fully_convolutional_networks(net, num_classes, is_training, dropout_keep_prob)
        else:
            net = fully_connected_networks(net, num_classes, is_training, dropout_keep_prob)
    return net


def fully_connected_networks(net, num_classes=2, is_training=True, dropout_keep_prob=0.5):
    # Use conv2d instead of fully_connected layers.
    net = layers.conv2d(net, 1024, net.get_shape()[1:3], padding='VALID')
    net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training)
    net = layers.conv2d(net, 1024, [1, 1])
    net = layers_lib.dropout(net, dropout_keep_prob, is_training=is_training)
    net = layers.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None)
    net = array_ops.squeeze(net, [1, 2])
    net = tf.nn.softmax(net, name='predicts')
    return net

def fully_convolutional_networks(net, num_classes=2, is_training=True, dropout_keep_prob=0.5):
    # Not use fully connected layers.
    net = layers.conv2d(net, 1024, [1, 1])
    net = layers.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None)
    net = tf.nn.softmax(net)
    net = tf.reduce_mean(net, [1, 2], name='predicts')
    return net
