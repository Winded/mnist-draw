import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import code

class MNISTSession(object):
    def __init__(self, mnist_data_path):
        self.mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)

        x = tf.placeholder(tf.float32, [None, 784])

        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        y = tf.nn.softmax(tf.matmul(x, W) + b)

        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        # This program creates initial values for our variable tensors. 
        # For W, this is two-dimensional (length 784x10) array of zeros. For b, this is one-dimensional (length 10) array of zeros.
        self.init = tf.global_variables_initializer()

        # This program evaluates our AIs accuracy
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Create a session
        self.session = tf.Session()
        # Run initialization. W and b are now initialized.
        self.session.run(self.init)

        self.x = x
        self.y = y
        self.y_ = y_

    def train(self):
        # Start spanking our AI.
        # Each time we run the training program, we give it 100 training samples from MNIST set.
        # To improve preformance, we use 100 samples at a time for optimization. We could get more accurate results if
        # we optimized once every MNIST sample, but that is computationally more expensive.
        for i in range(1000):
            batch_xs, batch_ys = self.mnist.train.next_batch(100)
            self.session.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

        # Evaluate our AIs accuracy after training. If everything went right, it should be 92% accurate. It is not precise, so we'll just make sure its above 90%
        accuracy = round(self.session.run(self.accuracy, feed_dict={self.x: self.mnist.test.images, self.y_: self.mnist.test.labels}) * 100)
        print("MNIST accuracy = %s" % accuracy)

    def guess_number(self, input):
        guess = self.session.run(tf.log(self.y), feed_dict={self.x: [input]})

        # Calculate average
        avg = 0
        for num in guess[0]:
            avg += num
        avg = avg / 10

        # Find max and it's index
        max_idx = -1
        max_num = -9999
        i = 0
        for num in guess[0]:
            if num > max_num:
                max_idx = i
                max_num = num
            i  += 1

        # Calculate max num offset from avg. If the difference is too low, we can't be certain that it is any number
        diff = abs(max_num - avg)
        if diff < 3 or max_idx == -1:
            return None

        return max_idx

    def correct_guess(self, input, correct_answer):
        # TODO
        pass

if __name__ == "__main__":
    mnist = MNISTSession("/home/tensorflow/MNIST_data/")
    mnist.train()

    print("Entering interactive mode..")
    code.interact(local=locals())