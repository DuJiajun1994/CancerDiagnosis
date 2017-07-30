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
from data_provider import DataProvider
from config_provider import get_config
from paths import Paths


def test_model(data_name, cfg_name, trained_model_name):
    cfg = get_config(cfg_name)
    print('Config:')
    print(cfg)
    input_data = DataProvider(data_name)

    meta_path = os.path.join(Paths.output_path, '{}.ckpt.meta'.format(trained_model_name))
    assert os.path.exists(meta_path), \
        '{} is not existed'.format(meta_path)
    saver = tf.train.import_meta_graph(meta_path)

    with tf.Session() as sess:
        # Load trained model
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        predicts = graph.get_tensor_by_name("predicts:0")

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
    parser.add_argument('--data', dest='data_name',
                        help='data to use',
                        default='', type=str)
    parser.add_argument('--cfg', dest='cfg_name',
                        help='train&test config to use',
                        default='', type=str)
    parser.add_argument('--net', dest='trained_model_name',
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
    test_model(args.data_name, args.cfg_name, args.trained_model_name)
