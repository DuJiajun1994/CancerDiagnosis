from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib import layers
slim = tf.contrib.slim

from models.vgg16 import vgg16, vgg16_fcn


def build_model(model_name, inputs, num_classes=2, is_training=True, dropout_keep_prob=0.5):
    if model_name == 'vgg16':
        return vgg16(inputs, num_classes, is_training, dropout_keep_prob)
    elif model_name == 'vgg16_fcn':
        return vgg16_fcn(inputs, num_classes, is_training, dropout_keep_prob)
    elif model_name.find('fcn') >= 0:
        return build_fcn_model(model_name[0, -4], inputs, num_classes, is_training, dropout_keep_prob)
    else:
        raise Exception('model {} is not existed'.format(model_name))

def build_fcn_model(model_name, inputs, num_classes=2, is_training=True, dropout_keep_prob=0.5):
    if model_name == 'inception_v1':
        net = inception.inception_v1_base(inputs)
    elif model_name == 'inception_v2':
        net = inception.inception_v2_base(inputs)
    elif model_name == 'inception_v2':
        net = inception.inception_v3_base(inputs)
    else:
        raise Exception('model {} is not existed'.format(model_name))

    net = layers.conv2d(net, 1024, [1, 1])
    net = layers.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None)
    net = tf.nn.softmax(net)
    net = tf.reduce_mean(net, [1, 2], name='predicts')
    return net