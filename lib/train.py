import tensorflow as tf
import os
import time
from datetime import datetime
import argparse
import sys
from model_provider import get_model
from data_provider import DataProvider
from config_provider import get_config

pretrain_model_path = 'data/pretrain_models/vgg16_pretrain_model'
snapshot_step = 1000


def train_model(model_name, data_name, cfg_name):
    cfg = get_config(cfg_name)

    model = get_model(model_name)
    input_data = DataProvider(data_name)

    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])  # images
    y = tf.placeholder(tf.float32, shape=[None])  # labels: 0, not cancer; 1, has cancer

    predict = model(x)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=cfg.learning_rate).minimize(loss)
    correct_predict = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Load pretrained model
        sess.run(saver.restore(sess, pretrain_model_path))

        print('Start training')
        train_loss = 0.
        train_accuracy = 0.
        for step in range(1, cfg.train_iters+1):
            images, labels = input_data.next_batch(cfg.batch_size, 'train')
            batch_loss, _, batch_accuracy = sess.run([loss, optimizer, accuracy],
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
            if step % snapshot_step == 0:
                timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
                save_path = os.path.join('output',
                                         '{}_{}_{}.txt'.format(data_name, model_name, timestamp))
                sess.run(saver.save(sess, save_path))

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
    parser.add_argument('--gpu', dest='gpu_id',
                        help='gpu id to use',
                        default=0, type=int)
    parser.add_argument('--net', dest='model_name',
                        help='net to use',
                        default='', type=str)
    parser.add_argument('--data', dest='data_name',
                        help='data to use',
                        default='vgg16', type=str)

    parser.add_argument('--cfg', dest='cfg_name',
                        help='cfg to use',
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
