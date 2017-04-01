
import tensorflow as tf
import numpy as np

# train set
# x_data의 값에 . 이 없으면 에러난다.

xy = np.loadtxt('./binary_data.csv', delimiter=',', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 반드시 tensorFlow가 지원하는 Variable로 해야 tf가 값을 변경할 수 있기 때문에 반드시 tf가 지원하는 Variable로 지정해야 한다.
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))


h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

# 손실 함수를 작성합니다.
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

learning_rate=tf.Variable(0.1)
# 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# 비용을 최소화 하는 것이 최종 목표
train_op = optimizer.minimize(cost)

# 세션을 생성하고 초기화합니다.
with tf.Session() as sess:
    # W, b 가 변수인데, 초기화를 안하면 초기화가 안되었다는 에러 메시지가 나타난다.
    # 세션에서 가장 먼저 실행시켜줘야 한다.
    sess.run(tf.global_variables_initializer())

    # 최적화를 100번 수행합니다.
    for step in range(2001):
        sess.run(train_op, feed_dict={X:x_data, Y:y_data})

        if step%20==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

    print("--------------------------------")

    # 공부한 시간, 출석
    print( sess.run(hypothesis, feed_dict={X:[[1], [2], [2]]}) >0.5)
    print( sess.run(hypothesis, feed_dict={X:[[1], [5], [5]]}) >0.5)
    print( sess.run(hypothesis, feed_dict={X:[[1, 1], [4, 3], [3, 5]]}) >0.5)