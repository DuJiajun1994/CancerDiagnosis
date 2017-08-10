import sys
import os
lib_path = os.getcwd()
print(lib_path)
sys.path.insert(0, lib_path)
print('System path:')
print(sys.path)

import tensorflow as tf
import time
from datetime import datetime
import argparse
import sys
from model_provider import get_model
from data_provider import DataProvider
from config_provider import get_config
from paths import Paths
import tensorflow.contrib.slim as slim


def get_pretrain_model_path(model_name):
    pretrain_model_name = model_name
    if model_name == 'vgg16_fcn':
        pretrain_model_name = 'vgg16'
    pretrain_model_path = os.path.join(Paths.data_path, 'pretrain_models', '{}.ckpt'.format(pretrain_model_name))
    assert os.path.exists(pretrain_model_path), \
        'pretrain model {} is not existed'.format(pretrain_model_path)
    return pretrain_model_path

def get_restore_vars(model_name):
    model_vals = slim.get_model_variables()
    print('Model vals:')
    for val in model_vals:
        print(val.op.name)

    restore_vals = []
    for val in model_vals:
        val_name = val.op.name
        if val_name.split('/')[1].find('conv') >= 0:
            restore_vals.append(val)
    print('Vals load from pretrained model:')
    for val in restore_vals:
        print(val.op.name)

    return restore_vals

def train_model(model_name, data_name, cfg_name):
    cfg = get_config(cfg_name)
    print('Config:')
    print(cfg)
    model = get_model(model_name)
    input_data = DataProvider(data_name)

    x = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.image_height, cfg.image_width, 3], name='x')  # images
    y = tf.placeholder(tf.int64, shape=[cfg.batch_size], name='y')  # labels: 0, not cancer; 1, has cancer
    labels = tf.one_hot(y, depth=2, on_value=1., off_value=0., dtype=tf.float32)
    predicts = model(x)
    loss = - tf.reduce_mean(tf.reduce_sum(labels * tf.log(predicts + 1e-10), 1))  # add 1e-10 to avoid log(0) = NaN
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=cfg.learning_rate).minimize(loss)
    correct_predict = tf.equal(tf.argmax(predicts, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name='accuracy')

    restore_vars = get_restore_vars(model_name)
    restorer = tf.train.Saver(restore_vars)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Load pretrained model
        pretrain_model_path = get_pretrain_model_path(model_name)
        restorer.restore(sess, pretrain_model_path)

        print('Start training')
        train_loss = 0.
        train_accuracy = 0.
        for step in range(1, cfg.train_iters+1):
            images, labels = input_data.next_batch(cfg.batch_size, 'train')
            batch_loss, _, batch_accuracy, batch_predict = sess.run([loss, optimizer, accuracy, predicts],
                                                        feed_dict={x: images, y: labels})
            train_loss += batch_loss
            train_accuracy += batch_accuracy
            # Display training status
            if step % cfg.display_step == 0:
                print("{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}"
                      .format(datetime.now(), step, train_loss / cfg.display_step, train_accuracy / cfg.display_step))
                train_loss = 0.
                train_accuracy = 0.

            # Snapshot
            if step % cfg.snapshot_step == 0:
                timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
                save_path = os.path.join('output',
                                         '{}_{}_{}'.format(data_name, model_name, step))
                saver.save(sess, save_path)

            # Display testing status
            if step % cfg.test_step == 0:
                test_accuracy = 0.
                test_num = int(input_data.test_size / cfg.batch_size)
                for _ in range(test_num):
                    images, labels = input_data.next_batch(cfg.batch_size, 'test')
                    acc = sess.run(accuracy, feed_dict={x: images, y: labels})
                    test_accuracy += acc
                test_accuracy /= test_num
                print("{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_accuracy))

        print('Finish!')


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train Cancer Diagnosis Network')
    parser.add_argument('--net', dest='model_name',
                        help='net to use',
                        default='vgg16', type=str)
    parser.add_argument('--data', dest='data_name',
                        help='data to use',
                        default='', type=str)
    parser.add_argument('--cfg', dest='cfg_name',
                        help='train&test config to use',
                        default='', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    train_model(args.model_name, args.data_name, args.cfg_name)
