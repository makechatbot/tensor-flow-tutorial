from random import randint

import tensorflow as tf
import numpy as np
import time


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 128
test_size = 2048


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    # X는 입력, w는 필터= 3x3x1, 32 output, strides는 1, 가로 이동거리, 세로이동거리, 1
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    print(l1a)

    # ksize 즉 필터가 2x2이고, strides의 이동거리가 2x2 이며, padding='SAME' 이므로 크기가 절반으로 나온다.
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))

    # 임시로 막아본다. 4 by 4 로 구분하기가 힘들 것 같아 7 by 7로 변형하여 FC(Fully Connected Network)에 입력 값으로 사용하도록 한다.
    # l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
    #                     strides=[1, 2, 2, 1], padding='SAME')

    l3=l3a


    # 입력의 크기가 점점 줄었다. 28x28 이었는데, 4x4까지 줄었다.
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img


with tf.name_scope('input'):
    X = tf.placeholder("float", [None, 28, 28, 1], name='X-Input')
    Y = tf.placeholder("float", [None, 10], name='Y-Input')

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
# 임시로 막아본다.
# w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
# 4 by 4 에서 7 by 7 로 입력되는 특징의 개수를 늘려본다.
w4 = init_weights([128 * 7 * 7, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

with tf.name_scope('weights'):
    tf.summary.histogram('weights1', w)
    tf.summary.histogram('weights2', w2)
    tf.summary.histogram('weights3', w3)
    tf.summary.histogram('weights4', w4)
    tf.summary.histogram('weights O', w_o)


p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.name_scope('weights'):
    tf.summary.histogram('activation', py_x)
    tf.summary.histogram('Y', Y)
    tf.summary.histogram('X', X)

with tf.name_scope('cost'):
    tf.summary.scalar('cost', cost)

spendTime = 0
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./log/mnist-cnn', sess.graph)


    for i in range(10):
        print(i)
        now = time.time()
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            print('.', end='', flush=True)
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        print('')


        spendTime+=time.time()-now
        print('second {}'.format(spendTime) )

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i,  np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
