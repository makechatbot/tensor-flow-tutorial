
import tensorflow as tf
import numpy as np

# train set
# x_data의 값에 . 이 없으면 에러난다.

xy = np.loadtxt('./grade_data.csv', delimiter=',', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])


X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 3])

# 3,3 ==> x가 3, y가 3
W = tf.Variable(tf.zeros([3, 3]))


hypothesis = tf.nn.softmax(tf.matmul(X, W))


learning_rate=0.001


# 손실 함수를 작성합니다.
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

# 텐서플로우에 기본적으로 포함되어 있는 함수를 이용해 경사 하강법 최적화를 수행합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 세션을 생성하고 초기화합니다.
with tf.Session() as sess:
    # W, b 가 변수인데, 초기화를 안하면 초기화가 안되었다는 에러 메시지가 나타난다.
    # 세션에서 가장 먼저 실행시켜줘야 한다.
    sess.run(tf.global_variables_initializer())

    # 최적화를 100번 수행합니다.
    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})

        if step%20==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

    print("--------------------------------")

    # 공부한 시간, 출석
    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print(a, sess.run(tf.arg_max(a, 1)))    # 위치를 돌려준다. 배열의 인덱스를 돌려준다.

    b = sess.run(hypothesis, feed_dict={X:[[1, 3, 4]]})
    print(b, sess.run(tf.arg_max(b, 1)))

    c = sess.run(hypothesis, feed_dict={X:[[1, 1, 0]]})
    print(c, sess.run(tf.arg_max(c, 1)))

    all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7],[1, 3, 4],[1, 1, 0]]})
    print(all, sess.run(tf.arg_max(all, 1)))
