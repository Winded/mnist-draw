import tensorflow as tf
import numpy as np

# Make 100 random X coordinate values
x_data = np.random.rand(100).astype(np.float32)
# Make 100 Y coordinate values with function y = x * 0.1 + 0.3
y_data = x_data * 0.2 + 0.1

# Initialize our weight tensor. This should end up as near 0.1
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# Initialize our bias tensor. This should end up as near 0.3
b = tf.Variable(tf.zeros([1]))
# Create tensor that calculates Y coordinate from weight and bias. 
# This is initially all kinds of wrong, but our neural network will learn the correct values
y = W * x_data + b

# Do a bunch of magic tensors that optimize our W and b variables to get correct results
# Tensor that calculates how wrong our AI is
loss = tf.reduce_mean(tf.square(y - y_data))

# Our training program. The optimizer backtraces the tensor network to find variables W and b, and when training
# is run, it optimizes those values closer to our correct results, found in y_data
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Training prgram flow:
# train -> loss -> reduce_mean -> square -> (y - y_data) -> (W * x_data + b)

# This program creates initial values for our variable tensors. 
# For W, this creates a random number from -1.0 to 1.0. For b, it sets it to zero.
init = tf.global_variables_initializer()

# Create a session
sess = tf.Session()
# Run initialization. W and b are now initialized.
sess.run(init)

print("Initial values: %f %f" % (sess.run(W), sess.run(b)))

# Start spanking our AI.
# Once every 20 steps, print our current values of W and b
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print("Step %i: %f %f" % (step, sess.run(W), sess.run(b)))