import math
import models
import utils
import numpy as np
import tensorflow as tf
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class BasicCnnModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def cnn(self, 
          model_input, 
          l2_penalty=1e-8, 
          num_filters = 64,
          filter_size = 3, 
          sub_scope="",
          **unused_params):
    batch_size, length, width, channel = model_input.get_shape().as_list()
    f = tf.get_variable(
            sub_scope+"_filter",
            shape=[filter_size, filter_size, channel, num_filters],
            dtype=tf.float32,
            regularizer=slim.l2_regularizer(l2_penalty))
    cnn_output = tf.nn.conv2d(model_input, filter=f, strides=[1,2,2,1], padding="SAME")
    return cnn_output

  def normalize(self, model_input):
    mean_input = tf.reduce_mean(model_input, axis=[1,2], keep_dims=True)
    std_input = tf.sqrt(tf.reduce_mean(tf.square(model_input - mean_input), axis=[1,2], keep_dims=True)) + 0.01
    output = (model_input - mean_input) / std_input
    return output

  def create_model(self, model_input, vocab_size,
                   dropout=False, keep_prob=None, 
                   l2_penalty=1e-8, sub_scope="", original_input=None, **unused_params):

    model_input = self.normalize(model_input)

    cnn_layer1 = self.cnn(model_input, num_filters=16, sub_scope="cnn_1", l2_penalty=l2_penalty)
    cnn_layer2 = self.cnn(cnn_layer1, num_filters=32, sub_scope="cnn_2", l2_penalty=l2_penalty)
    cnn_layer3 = self.cnn(cnn_layer2, num_filters=64, sub_scope="cnn_3", l2_penalty=l2_penalty)
    cnn_layer4 = self.cnn(cnn_layer3, num_filters=64, sub_scope="cnn_4", l2_penalty=l2_penalty)
    cnn_layer5 = self.cnn(cnn_layer4, num_filters=64, sub_scope="cnn_5", l2_penalty=l2_penalty)
    cnn_layer6 = self.cnn(cnn_layer5, num_filters=64, sub_scope="cnn_6", l2_penalty=l2_penalty)

    fc_layer1 = slim.fully_connected(
        tf.reshape(cnn_layer6, [-1, 1024]), 
        256,
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="fc_1")

    predictions = slim.fully_connected(
        fc_layer1, 
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="output")
    return {"predictions": predictions}

