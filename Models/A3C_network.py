import tensorflow as tf

from keras.layers import *


class A3C_network():

    def __init__(self, name, n_timesteps, height, width, depth, output_dim,
                 cnn_layers, a_filters, a_strides, dropouts, kernel_sizes,
                 rnn_dropouts,
                 saving_path, check_point=False
                 ):
        """Network structure is defined here

        Args:
            name (str): The name of scope
            input_shape (list): The shape of input image [H, W, C]
            output_dim (int): Number of actions
        """
        self.states = tf.placeholder(tf.float32, shape=[None, n_timesteps, height, width, depth], name="states")
        self.actions = tf.placeholder(tf.uint8, shape=[None, height, width, depth], name="actions")
        self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
        self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")

        action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
        net = self.states

        for cnn_layer in range(cnn_layers):
            with tf.variable_scope("layer%i" % cnn_layer):
                net = ConvLSTM2D(filters=self.a_filters[cnn_layer],
                                 kernel_size=self.kernel_sizes[cnn_layer],
                                 strides=[1, 1],
                                 padding='same',
                                 dropout=self.dropout[cnn_layer],
                                 input_shape=(None, self.weight, self.height, self.depth),
                                 return_sequences=True,
                                 recurrent_dropout=self.rnn_dropout[cnn_layer])

                net = tf.nn.relu(net, name="relu")

        with tf.variable_scope("fc1"):
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 256, name='dense')
            net = tf.nn.relu(net, name='relu')

        # actor network
        action_values = tf.layers.dense(net, output_dim, name="final_fc")
        # self.action_prob = tf.nn.softmax(action_values, name="action_prob")
        # single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)

        entropy = - self.action_prob * tf.log(self.action_prob + 1e-7)
        entropy = tf.reduce_sum(entropy, axis=1)

        log_action_prob = tf.log(single_action_prob + 1e-7)
        maximize_objective = log_action_prob * self.advantage + entropy * 0.005
        self.actor_loss = - tf.reduce_sum(maximize_objective)

        # value network
        self.values = tf.squeeze(tf.layers.dense(net, 1, name="values"))
        self.value_loss = tf.reduce_sum(tf.squared_difference(self.rewards, self.values))

        self.total_loss = self.actor_loss + self.value_loss * .5
        global_step = tf.contrib.framework.get_or_create_global_step()
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=.99)
        self.train_op = self.optimizer.minimize(self.total_loss, global_step=global_step)

        loss_summary = tf.summary.scalar("total_loss", self.total_loss)
        value_summary = tf.summary.histogram("values", self.values)

        self.summary_op = tf.summary.merge([loss_summary, value_summary])
