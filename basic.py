import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # silences the TF warnings!

tf.compat.v1.disable_eager_execution()  # disable eager execution

a = 2
b = 3
c = tf.add(a, b, name='Add')
print(c)

sess = tf.compat.v1.Session()
print(sess.run(c))
sess.close()