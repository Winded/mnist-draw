import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import code

# These implementations mostly follow TensorFlow tutorials
# https://www.tensorflow.org/get_started/mnist/beginners
# https://www.tensorflow.org/get_started/mnist/pros

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
        guess = self.session.run(self.y, feed_dict={self.x: [input]})
        guess = guess[0]

        # Get maximum value and index
        max_idx = -1
        max_num = 0
        i = 0
        while i < 10:
            if guess[i] > max_num:
                max_num = guess[i]
                max_idx = i
            i += 1

        # If our probabilities are not high enough or, for some reason, our index is undefined, 
        # we were unable to determine the number, so we return None.
        if max_num < 0.3 or max_idx < 0:
            return None

        return max_idx

    def correct_guess(self, input, correct_answer):
        if type(correct_answer) != int or correct_answer < 0 or correct_answer > 9:
            return False
        if len(input) != 784:
            return False

        correct_y = np.zeros([10])
        correct_y[correct_answer] = 1.0

        self.session.run(self.train_step, feed_dict={self.x: [input], self.y_: correct_y})
        return True

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class MNISTConvSession(object):
    def __init__(self, mnist_data_path):
        self.mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)

        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(x, [-1,28,28,1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.init = tf.global_variables_initializer()

        self.train_step = train_step
        self.accuracy = accuracy
        self.keep_prob = keep_prob

        self.x = x
        self.y = y_conv
        self.y_ = y_

        self.session = tf.Session()
        self.session.run(self.init)

    def train(self):
        for i in range(20000):
            batch = self.mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = self.accuracy.eval(session=self.session, feed_dict={
                    self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            self.train_step.run(session=self.session, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
        print("MNIST accuracy = %g" % self.accuracy.eval(session=self.session, feed_dict={x: self.mnist.test.images, y_: self.mnist.test.labels, keep_prob: 1.0}))

    def guess_number(self, input):
        guess = self.session.run(self.y, feed_dict={self.x: [input]})
        guess = guess[0]

        # Get maximum value and index
        max_idx = -1
        max_num = 0
        i = 0
        while i < 10:
            if guess[i] > max_num:
                max_num = guess[i]
                max_idx = i
            i += 1

        # If our probabilities are not high enough or, for some reason, our index is undefined, 
        # we were unable to determine the number, so we return None.
        if max_num < 0.3 or max_idx < 0:
            return None

        return max_idx

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.session, "/home/tensorflow/model.ckpt")

if __name__ == "__main__":
    mnist = MNISTSession("/home/tensorflow/MNIST_data/")
    mnist.train()

    mnist2 = MNISTConvSession("/home/tensorflow/MNIST_data/")
    try:
        mnist2.train()
    except:
        pass
    mnist2.save()

    print("Entering interactive mode..")
    code.interact(local=locals())