import tensorflow as tf
import numpy as np

# train set
# x_data의 값에 . 이 없으면 에러난다.



xy = np.loadtxt('./xor_data.csv', delimiter=',', unpack=True, dtype='float32')

# x_data = xy[0:-1]
# y_data = xy[-1]
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))

print('----- xdata')
print(x_data)
print('----- ydata')
print(y_data)

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, name='X-input')
    Y = tf.placeholder(tf.float32, name='Y-input')

# 반드시 tensorFlow가 지원하는 Variable로 해야 tf가 값을 변경할 수 있기 때문에 반드시 tf가 지원하는 Variable로 지정해야 한다.
print('len(x_data) = ',len(x_data))

# W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

# W1 = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))
# W2 = tf.Variable(tf.random_uniform([len(x_data), 1], -1.0, 1.0))

l1_nn = 2

W1 = tf.Variable(tf.random_uniform([2, l1_nn], -1.0, 1.0), name='Weight1')
W2 = tf.Variable(tf.random_uniform([l1_nn, 1], -1.0, 1.0), name='Weight2')

with tf.name_scope('weights'):
    tf.summary.histogram('weights1', W1)
    tf.summary.histogram('weights2', W2)

b1 = tf.Variable(tf.zeros([l1_nn]), name="Bias1")
# b1 = tf.Variable(tf.zeros([len(x_data)]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")


with tf.name_scope('biase'):
    tf.summary.histogram('biasses1', b1)
    tf.summary.histogram('biasses2', b2)

# h = tf.matmul(X, W)
# L2 = tf.sigmoid(tf.matmul(W1, X) + b1)
# hypothesis = tf.sigmoid(tf.matmul(W2, L2) + b2)
L2 = tf.nn.relu(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)
# hypothesis = tf.sigmoid(h)

with tf.name_scope('weights'):
    tf.summary.histogram('hypothesis', hypothesis)
    tf.summary.histogram('Y', Y)
    tf.summary.histogram('X', X)
# 손실 함수를 작성합니다.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis), name='cost')

with tf.name_scope('cost'):
    tf.summary.scalar('cost', cost)
a = tf.Variable(0.01)
# 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행합니다.
optimizer = tf.train.GradientDescentOptimizer(a)
# 비용을 최소화 하는 것이 최종 목표
train = optimizer.minimize(cost, name='train')



# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()


# 세션을 생성하고 초기화합니다.
with tf.Session() as sess:
    # W, b 가 변수인데, 초기화를 안하면 초기화가 안되었다는 에러 메시지가 나타난다.
    # 세션에서 가장 먼저 실행시켜줘야 한다.
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./log/xor_logs', sess.graph)


    with tf.name_scope('cost'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('accuracy', accuracy)
    # 최적화를 100번 수행합니다.
    for step in range(2001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})

        with tf.name_scope('cost'):
            if step % 100 ==0:
                train_writer.add_summary(sess.run(merged, feed_dict={X: x_data, Y: y_data}), step)
        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W1))

    print("--------------------------------")



    print(sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data}))

    print("Accuracy : ", accuracy.eval({X:x_data, Y:y_data}))

    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('./log/xor_logs', sess.graph)
    tf.global_variables_initializer().run()



    # # 공부한 시간, 출석
    # print(sess.run(hypothesis, feed_dict={X: [[0],[0]]}) > 0.5)
    # print(sess.run(hypothesis, feed_dict={X: [[0],[1]]}) > 0.5)
    # print(sess.run(hypothesis, feed_dict={X: [[1],[0]]}) > 0.5)
    # print(sess.run(hypothesis, feed_dict={X: [[1],[1]]}) > 0.5)
    # # print(sess.run(hypothesis, feed_dict={X: [[0,0]]}) > 0.5)
    # # print(sess.run(hypothesis, feed_dict={X: [[0,1]]}) > 0.5)
    # # print(sess.run(hypothesis, feed_dict={X: [[1,0]]}) > 0.5)
    # # print(sess.run(hypothesis, feed_dict={X: [[1,1]]}) > 0.5)
