import tensorflow as tf
import os
import time
from datetime import datetime
from model_provider import get_model
from data_provider import DataProvider

pretrain_model_path = 'data/pretrain_models/vgg16_pretrain_model'
snapshot_step = 1000


def train_model(model_name, data_name, train_iters, test_step, learning_rate, batch_size, display_step=20):
    model = get_model(model_name)
    input_data = DataProvider(data_name)

    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])  # images
    y = tf.placeholder(tf.float32, shape=[None])  # labels: 0, not cancer; 1, has cancer

    predict = model(x)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_predict = tf.equal(tf.argmax(predict, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Load pretrained model
        sess.run(saver.restore(sess, pretrain_model_path))

        print('Start training')
        for step in range(1, train_iters+1):
            images, labels = input_data.next_batch(batch_size, 'train')
            train_loss, _, train_accuracy = sess.run([loss, optimizer, accuracy],
                                                     feed_dict={x: images, y: labels})

            # Display training status
            if step % display_step == 0:
                print("{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}"
                      .format(datetime.now(), step, train_loss, train_accuracy))

            # Snapshot
            if step % snapshot_step == 0:
                timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
                save_path = os.path.join('output',
                                         '{}_{}_{}.txt'.format(data_name, model_name, timestamp))
                sess.run(saver.save(sess, save_path))

            # Display testing status
            if step % test_step == 0:
                test_accuracy = 0.
                test_num = int(input_data.test_size / batch_size)
                for _ in range(test_num):
                    images, labels = input_data.next_batch(batch_size, 'test')
                    acc = sess.run(accuracy, feed_dict={x: images, y: labels})
                    test_accuracy += acc
                test_accuracy /= test_num
                print("{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_accuracy))

        print('Finish!')
