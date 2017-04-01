# X 와 Y 의 상관관계를 분석하는 기초적인 선형 회귀 모델을 만들고 실행해봅니다.

import tensorflow as tf

# train set
x_data = [1, 2, 3, 4]
y_data = [1, 2, 3, 4]

# 반드시 tensorFlow가 지원하는 Variable로 해야 tf가 값을 변경할 수 있기 때문에 반드시 tf가 지원하는 Variable로 지정해야 한다.
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# X 와 Y 의 상관 관계를 분석하기 위한 가설 수식을 작성합니다.
# h = W * X + b
# W 와 X 가 행열이 아니므로 tf.matmul 이 아니라 tf.mul 을 사용했습니다.
hypothesis = W*X+b

# 손실 함수를 작성합니다.
# mean(h - Y)^2 : 예측값과 실제값의 거리를 비용(손실) 함수로 정합니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))
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
    for step in range(4001):
        # sess.run 을 통해 train_op 와 cost 그래프를 계산합니다.
        # 이 때, 가설 수식에 넣어야 할 실제값을 feed_dict 을 통해 전달합니다.
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        if step%5==0:
            print(step, cost_val, sess.run(W), sess.run(b))

    print("\n=== Test ===")
    # 최적화가 완료된 모델에 테스트 값을 넣고 결과가 잘 나오는지 확인해봅니다.
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))


