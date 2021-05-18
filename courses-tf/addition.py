import tensorflow.compat.v1 as tf
import os
import numpy as np

tf.disable_v2_behavior()

print(tf.version.VERSION)

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define computational graph
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

addition = tf.add(X, Y, name="addition")

# create the session to execute computational graph
with tf.compat.v1.Session() as sess:
    result = sess.run(addition, feed_dict={X: [1, 2, 10], Y: [4, 2, 10]})
    print(result)
