import tensorflow as tf
import numpy as np

class DQN():

    def __init__(self, state_size, action_size, learning_rate, n_frames, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name, reuse=False):

            self.state = tf.placeholder(tf.float32, [None, 1, state_size, n_frames], name='state')
            self.actions = tf.placeholder(tf.float32, [None, action_size], name='actions')

            self.target_Q = tf.placeholder(tf.float32, [None], name='target')

            self.conv1 = tf.layers.conv2d(inputs = self.state,
                                          filters = 16,
                                          kernel_size = [1, 1],
                                          strides = [4, 4],
                                          padding = 'VALID',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name = 'conv1',
                                          reuse = False)

            self.conv1_out = tf.nn.elu(self.conv1, name='conv1_out')

            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                          filters = 32,
                                          kernel_size = [1, 1],
                                          strides = [4, 4],
                                          padding = 'VALID',
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name = 'conv2')

            self.conv2_out = tf.nn.elu(self.conv2, name='conv2_out')

            self.flatten = tf.contrib.layers.flatten(self.conv2_out)

            self.fc = tf.layers.dense(inputs = self.flatten,
                                      units = 128,
                                      activation = tf.nn.elu,
                                      kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                      name = 'fc1')

            self.output = tf.layers.dense(inputs = self.fc,
                                          kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                          units = self.action_size,
                                          activation = None)

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions))

            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train(self, states, actions, target_Q, view_loss=False):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.optimizer, feed_dict = {self.state: states,
                                                    self.actions: actions,
                                                    self.target_Q: target_Q})

            if view_loss:
                loss = sess.run(self.loss, feed_dict = {self.state: states,
                                                        self.actions: actions,
                                                        self.target_Q: target_Q})
                print('loss {}'.format(str(loss)))

    def get_max_Q(self, states, batch_size):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            Q_values = sess.run(self.output, feed_dict = {self.state: states})
            max_Q_values_indices = np.argmax(Q_values, axis=1)
            max_Q_values = np.array([Q_values[j][max_Q_values_indices[j]] for j in range(batch_size)])
            return max_Q_values
