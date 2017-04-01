
import tensorflow as tf
import numpy as np

# train set
# x_data의 값에 . 이 없으면 에러난다.

xy = np.loadtxt('./linear_data.csv', delimiter=',', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

# 반드시 tensorFlow가 지원하는 Variable로 해야 tf가 값을 변경할 수 있기 때문에 반드시 tf가 지원하는 Variable로 지정해야 한다.
W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))


# X 와 Y 의 상관 관계를 분석하기 위한 가설 수식을 작성합니다.
# h = W * X + b
# W 와 X 가 행열이므로 tf.matmul 을 사용했습니다.
hypothesis = tf.matmul(W, x_data)

# 손실 함수를 작성합니다.
# mean(h - Y)^2 : 예측값과 실제값의 거리를 비용(손실) 함수로 정합니다.
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
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
        sess.run(train_op)

        if step%20==0:
            print(step, sess.run(cost), sess.run(W))
