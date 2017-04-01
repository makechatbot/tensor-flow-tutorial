import tensorflow as tf
import numpy as np

# train set
# x_data의 값에 . 이 없으면 에러난다.

xy = np.loadtxt('./xor_data.csv', delimiter=',', unpack=True, dtype='float32')


x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

lenX_data = len(x_data)
W1 = tf.Variable(tf.random_uniform([2, 4], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([4, 4], -1.0, 1.0))


b1 = tf.Variable(tf.zeros([4]), name="Bias1")
b2 = tf.Variable(tf.zeros([4]), name="Bias1")
#
L1 = tf.sigmoid(tf.matmul(X, W1)+b1)
hypothesis = tf.sigmoid((tf.matmul(L1, W2)+b2))
# hypothesis = tf.div(1., 1. + tf.exp(-tf.matmul(X, W1)))

# 손실 함수를 작성합니다. - 크로스 엔트로피 함수.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

learning_rate = tf.Variable(0.1)
# 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# 비용을 최소화 하는 것이 최종 목표
train = optimizer.minimize(cost)

# 세션을 생성하고 초기화합니다.
with tf.Session() as sess:
    # W, b 가 변수인데, 초기화를 안하면 초기화가 안되었다는 에러 메시지가 나타난다.
    # 세션에서 가장 먼저 실행시켜줘야 한다.
    sess.run(tf.global_variables_initializer())

    # 최적화를 100번 수행합니다.
    for step in range(1001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})

        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    print("--------------------------------")

    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(correct_prediction, feed_dict={X:x_data, Y:y_data}))
    print(sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data}))
    print("Accruracy : ", accuracy.eval({X: x_data, Y: y_data}))
