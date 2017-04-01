from random import randint

import tensorflow as tf
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs+n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0/(n_inputs+n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='X-Input') # 이미지 픽셀 28*28=784
    y = tf.placeholder(tf.float32, [None, 10], name='Y-Input') # 0-9 니까 총 10개.

# Set model weights
# W1 = tf.Variable(tf.random_normal([784, 256]), name='Weight1')
# W2 = tf.Variable(tf.random_normal([256, 256]), name='Weight2')
# W3 = tf.Variable(tf.random_normal([256, 10]), name='Weight3')

W1 = tf.get_variable("W1", shape=[784, 256], initializer=xavier_init(784, 256))
W2 = tf.get_variable("W2", shape=[256, 256], initializer=xavier_init(256, 256))
W3 = tf.get_variable("W3", shape=[256, 256], initializer=xavier_init(256, 256))
W4 = tf.get_variable("W4", shape=[256, 256], initializer=xavier_init(256, 256))
W5 = tf.get_variable("W5", shape=[256, 10], initializer=xavier_init(256, 10))


with tf.name_scope('weights'):
    tf.summary.histogram('weights1', W1)
    tf.summary.histogram('weights2', W2)
    tf.summary.histogram('weights3', W3)
    tf.summary.histogram('weights4', W4)
    tf.summary.histogram('weights5', W5)

b1 = tf.Variable(tf.random_normal([256]), name="Bias1")
b2 = tf.Variable(tf.random_normal([256]), name="Bias2")
b3 = tf.Variable(tf.random_normal([256]), name="Bias3")
b4 = tf.Variable(tf.random_normal([256]), name="Bias4")
b5 = tf.Variable(tf.random_normal([10]), name="Bias5")


with tf.name_scope('biase'):
    tf.summary.histogram('biasses1', b1)
    tf.summary.histogram('biasses2', b2)
    tf.summary.histogram('biasses3', b3)
    tf.summary.histogram('biasses4', b4)
    tf.summary.histogram('biasses5', b5)


dropout_rate = tf.placeholder(tf.float32)


# Construct model
_L1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))
L4 = tf.nn.dropout(_L4, dropout_rate)


# activation = tf.add(tf.matmul(L2, W3), b3)
activation = tf.matmul(L4, W5)+ b5



with tf.name_scope('weights'):
    tf.summary.histogram('activation', activation)
    tf.summary.histogram('Y', y)
    tf.summary.histogram('X', x)

# Minimize error using "cross entropy"
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=activation, labels=y))


with tf.name_scope('cost'):
    tf.summary.scalar('cost', cost)

# Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Softmax loss

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter('./log/mnist-dropout', sess.graph)


    with tf.name_scope('cost'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()

    # Training cycle - 데이터가 많으니까 쪼개서 학습한다.
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys, dropout_rate:0.7})
            # Compute average loss
            avg_cost += c / total_batch
            with tf.name_scope('cost'):
                if i % 100 ==0:
                    train_writer.add_summary(sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, dropout_rate:0.7}), i)
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    # correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels, dropout_rate:1.}))

    r = randint(0, mnist.test.num_examples - 1)
    print("Label : ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(activation, 1), {x: mnist.test.images[r:r + 1], dropout_rate:1.}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()