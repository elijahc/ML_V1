import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class agent():
    def __init__(self, lr, s_size, a_size)
        # Agent takes a state and produces an action
        self.state_in = tf.placeholder(shape=[1], dtype=tf.int32)

        # Onehot of state
        state_in_OH = slim.one_hot_encoding(self.state_in,s_size)
        output = slim.full_connected(state_in_OH,a_size,
                biases_initializer=None,activation_fn=tf.nn.sigmoid,weights_initializer=tf.ones)
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output,0)
