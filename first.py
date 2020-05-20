# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # gets rid of weird warning
import tensorflow as tf
print(tf.version)  # make sure the version is 2.x

# setup for a few tensors
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

# ranks/degrees/shape
rank1_tensor = tf.Variable(["Test"], tf.string)
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

print(tf.rank(rank2_tensor))
print(rank2_tensor.shape)

print("SEPARATE")

# changing shape of tensors!

tensor1 = tf.ones([1, 2, 3])  # tf.ones() creates a shape [1,2,3] tensor full of ones
tensor2 = tf.reshape(tensor1, [2, 3, 1])  # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
# this will reshape the tensor to [3,3]

# The numer of elements in the reshaped tensor MUST match the number in the original


print(tensor1)
print(tensor2)
print(tensor3)
# Notice the changes in shape

