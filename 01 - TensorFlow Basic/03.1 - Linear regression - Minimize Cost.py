
import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# y = Wx 로 식을 단순화
hypothesis = W * X

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# W 값을 재계산
descent = W - tf.multiply(0.1, tf.reduce_mean(tf.multiply((tf.multiply(W, X) - Y), X)))

# 업데이트 된 W 값을 재할당
update = W.assign(descent)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(200):
    sess.run(update, feed_dict={X:x_data, Y:y_data})
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
    if sess.run(cost, feed_dict={X:x_data, Y:y_data}) == 0:
        break

print('실제 값이 어떻게 나오나 보자 = ', sess.run(hypothesis, feed_dict={X:5}))
print('실제 값이 어떻게 나오나 보자 = ', sess.run(hypothesis, feed_dict={X:50}))