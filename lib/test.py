import sys
import os
lib_path = os.getcwd()
print(lib_path)
sys.path.insert(0, lib_path)
print('System path:')
print(sys.path)

import tensorflow as tf
import argparse
import sys
from model_provider import get_model
from data_provider import DataProvider
from config_provider import get_config
from paths import Paths


def get_trained_model_path(model_name):
    model_path = os.path.join(Paths.output_path, '{}.ckpt'.format(model_name))
    assert os.path.exists(model_path), \
        'trained model {} is not existed'.format(model_path)
    return model_path


def test_model(model_name, data_name, cfg_name, trained_model_name):
    cfg = get_config(cfg_name)
    print('Config:')
    print(cfg)
    model = get_model(model_name)
    input_data = DataProvider(data_name)

    x = tf.placeholder(tf.float32, shape=[cfg.batch_size, cfg.image_height, cfg.image_width, 3])  # images
    y = tf.placeholder(tf.int64, shape=[cfg.batch_size])  # labels: 0, not cancer; 1, has cancer
    predicts = model(x)
    correct_predict = tf.equal(tf.argmax(predicts, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Load trained model
        model_path = get_trained_model_path(trained_model_name)
        saver.restore(sess, model_path)

        test_accuracy = 0.
        test_num = int(input_data.test_size / cfg.batch_size)
        for _ in range(test_num):
            images, labels = input_data.next_batch(cfg.batch_size, 'test')
            acc, pred = sess.run([accuracy, predicts], feed_dict={x: images, y: labels})
            test_accuracy += acc
            for i in range(cfg.batch_size):
                print('{} {}'.format(labels[i], pred[i]))
        test_accuracy /= test_num
        print("Testing Accuracy = {:.4f}".format(test_accuracy))
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
    parser.add_argument('--var', dest='trained_model_name',
                        help='trained model to use',
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
    test_model(args.model_name, args.data_name, args.cfg_name, args.trained_model_name)
