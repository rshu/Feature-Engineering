import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
print(tf.version.VERSION)

# Build the graph
A = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
x = tf.constant([[0, 10], [0, 0.5]])
b = tf.constant([[1, -1]], dtype=tf.float32)
# y = Ax + b
y = tf.add(tf.matmul(A, x), b, name="result")

writer = tf.summary.FileWriter('log/matmul', tf.get_default_graph())
writer.close()