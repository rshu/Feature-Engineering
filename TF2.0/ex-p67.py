import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

print(tf.version.VERSION)

A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
x = tf.constant([[0, 10], [0, 0.5]])
b = tf.constant([[1, -1]], dtype=tf.float32)
y = tf.add(tf.matmul(A, x), b, name="result")

writer = tf.summary.FileWriter("log/matmul", tf.get_default_graph())
writer.close()

with tf.compat.v1.Session() as sess:
    A_value, x_value, b_value = sess.run([A, x, b])
    y_value = sess.run(y)

    # overwrite
    # feed_dict 可以填充 tf.placeholder
    # 也可以重写节点的值，但是数据类型和形状要相同
    y_new = sess.run(y, feed_dict={b: np.zeros((1, 2))})

print(f"A: {A_value}\nx: {x_value}\nb: {b_value}\n\ny: {y_value}")
print(f"y_new: {y_new}")
