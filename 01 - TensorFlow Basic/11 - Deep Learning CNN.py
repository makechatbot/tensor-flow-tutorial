from random import randint

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Parameters
learning_rate = 0.001

x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X-Input') # 이미지 픽셀 28*28=784
y = tf.placeholder(tf.float32, [None, 10], name='Y-Input') # 0-9 니까 총 10개.

output1 = 32
output2 = 64
output3 = 128
output4 = 625

dropout_rate = tf.placeholder(tf.float32)

def init_weights(weights):
    return tf.Variable(tf.random_normal(weights, stddev=0.01))

W1 = init_weights([3, 3, 1, output1])
W2 = init_weights([3, 3, output1, output2])
W3 = init_weights([3, 3, output2, output3])
W4 = init_weights([128 * 4 * 4, output4])
w_o = init_weights([output4, 10])


def init_conv2d(input, filter, weight):
    return tf.nn.relu(tf.nn.conv2d(input, weight, strides=filter, padding='SAME'))

def init_maxPool(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def init_dropOut(pool):
    return tf.nn.dropout(pool, dropout_rate)


l1a = init_conv2d(x, [1, 1, 1, 1], W1)
print(l1a)
pool1 = init_maxPool(l1a)
print(pool1)
pool1 = init_dropOut(pool1)
print(pool1)

l2a = init_conv2d(pool1, [1, 1, 1, 1], W2)
print(l2a)
pool2 = init_maxPool(l2a)
print(pool2)
pool2 = init_dropOut(pool2)
print(pool2)


l3a = init_conv2d(pool2, [1, 1, 1, 1], W3)
print(l3a)
pool3 = init_maxPool(l3a)
print(pool3)
pool3 = tf.reshape(pool3, [-1, W4.get_shape().as_list()[0]])
print(pool3)
pool3 = init_dropOut(pool3)
print(pool3)

l4 = tf.nn.relu(tf.matmul(pool3, W4))
l4 = init_dropOut(l4)

activation = tf.matmul(l4, w_o)


# Minimize error using "cross entropy"
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=activation, labels=y))

# Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Softmax loss

# Initializing the variables
init = tf.global_variables_initializer()


batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)


        _, cost_val = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys, dropout_rate:0.5})

        if i % 100 == 0:
            print(cost_val)


    check_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
    print('정확도:', sess.run(accuracy,
                           feed_dict={x: mnist.test.images.reshape(-1, 28, 28, 1),
                                      y: mnist.test.labels, dropout_rate:1.}))