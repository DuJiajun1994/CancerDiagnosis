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

    x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
    y = tf.placeholder(tf.float32, [batch_size, 2])

    pred = model(x)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Load pretrained model
        sess.run(saver.restore(sess, pretrain_model_path))

        print('Start training')
        for step in range(1, train_iters+1):
            batch_xs, batch_ys = input_data.next_batch(batch_size, 'train')
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

            # Display testing status
            if step % test_step == 0:
                test_acc = 0.
                test_count = 0
                for _ in range(int(input_data.test_size / batch_size)):
                    batch_tx, batch_ty = input_data.next_batch(batch_size, 'test')
                    acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc))

            # Display training status
            if step % display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
                batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
                print("{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}"
                      .format(datetime.now(), step, batch_loss, acc))

            if step % snapshot_step == 0:
                timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
                save_path = os.path.join('output',
                                         '{}_{}_{}.txt'.format(data_name, model_name, timestamp))
                sess.run(saver.save(sess, save_path))

        print('Finish!')