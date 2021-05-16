import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
print(tf.version.VERSION)

with tf.device("/CPU:0"):
    A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    x = tf.constant([[0, 10], [0, 0.5]])
    b = tf.constant([[1, -1]], dtype=tf.float32)

with tf.device("/GPU:0"):
    mul = A @ x

writer = tf.summary.FileWriter("log/matmul_optimized", tf.get_default_graph())
writer.close()
