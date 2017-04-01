
import tensorflow as tf
# sudo pip3 install matplotlib
import matplotlib.pyplot as plt

# tf 그래프 Input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samles = len(X)

# Set model weights
W = tf.placeholder(tf.float32)

# linear model 생성
hypothesis = tf.multiply(X, W)

# cost 함수
cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2))/(m)

# placeholder가 있으므로 변수 초기화
init = tf.initialize_all_variables()

# 그래프를 위해 저장.
W_val = []
cost_val = []

# 그래프를 데이타 넣기.
sess = tf.Session()
sess.run(init)

for i in range(-30, 50):
    print(i*0.1, sess.run(cost, feed_dict={W: i*0.1}))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))

# 그래프 show
plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()