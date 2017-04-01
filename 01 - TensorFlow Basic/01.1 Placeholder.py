
import tensorflow as tf

"""
placeholder 는 변수의 타입만 정한다.
"""

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

"""
operation에 타입만 정해서 넘겨서 모델을 생성한다.
용어가 동일한 것이 다르게 표현되는 것이 많은데,
정리하자면

Node=Operation=Model은 동의어이다.
"""
add = tf.add(a, b)
mul = tf.multiply(a, b)

"""
실제 operation이 동작할 때 값을 넘긴다.
"""
with tf.Session() as sess:
    print("Addition with variables: %i " % sess.run(add, feed_dict={a:2, b:3}))
    print("Multiplication with variables : %i" % sess.run(mul, feed_dict={a:2, b:3}))



"""
미리 타입을 2단계에 걸쳐서 정의하는 것이네.
"""