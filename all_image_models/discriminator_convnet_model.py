import math
import models
import utils
import numpy as np
import tensorflow as tf
from tensorflow import flags
import tensorflow.contrib.slim as slim
FLAGS = flags.FLAGS

class DiscriminatorConvNetModel(models.BaseModel):
  """Build a U-Net architecture"""

  def conv_conv_pool(self, input_, n_filters, is_training, name, pool=True, activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        is_training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
      for i, F in enumerate(n_filters):
        net = tf.layers.conv2d(net, F, (3, 3), activation=None, padding='same', name="conv_{}".format(i + 1))
        net = tf.layers.batch_normalization(net, training=is_training, name="bn_{}".format(i + 1))
        net = activation(net, name="relu{}_{}".format(name, i + 1))

      if pool is False:
        return net

      pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

      return net, pool


  def upsample_concat(self, inputA, input_B, name):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    upsample = self.upsampling_2D(inputA, size=(2, 2), name=name)

    return tf.concat([upsample, input_B], axis=-1, name="concat_{}".format(name))


  def downsampling_2D(self, tensor, name, size=(2, 2)):
    """Downsample/Rescale `tensor` by size
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2))
    Returns:
        output (4-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    H, W = tensor.get_shape().as_list()[1:3]

    H_denom, W_denom = size
    target_H = H / H_denom
    target_W = W / W_denom

    return tf.image.resize_images(tensor, (target_H, target_W))

  def upsampling_2D(self, tensor, name, size=(2, 2)):
    """Upsample/Rescale `tensor` by size
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        name (str): name of upsampling operations
        size (tuple, optional): (height_multiplier, width_multiplier)
            (default: (2, 2))
    Returns:
        output (4-D Tensor): (N, h_multiplier * H, w_multiplier * W, C)
    """
    H, W, _ = tensor.get_shape().as_list()[1:]

    H_multi, W_multi = size
    target_H = H * H_multi
    target_W = W * W_multi

    return tf.image.resize_nearest_neighbor(tensor, (target_H, target_W), name="upsample_{}".format(name))

  def convnet(self, model_input, is_training=True, **unused_params):

    print "model_input", model_input
    float_input = tf.cast(model_input, dtype=tf.float32)
    print "float_input", float_input
    scaled_input = (1.0 / 127.5) * float_input - 1
    print "scaled_input", scaled_input
    net = tf.layers.conv2d(scaled_input, 3, (1, 1), name="color_space_adjust")
    print "net", net
    conv1, pool1 = self.conv_conv_pool(net, [8, 8], is_training, name=1)
    print "conv1", conv1
    print "pool1", pool1
    conv2, pool2 = self.conv_conv_pool(pool1, [16, 16], is_training, name=2)
    print "conv2", conv2
    print "pool2", pool2
    conv3, pool3 = self.conv_conv_pool(pool2, [32, 32], is_training, name=3)
    print "conv3", conv3
    print "pool3", pool3
    conv4, pool4 = self.conv_conv_pool(pool3, [32, 32], is_training, name=4)
    print "conv4", conv4
    print "pool4", pool4
    conv5 = self.conv_conv_pool(pool4, [16, 16], is_training, name=5, pool=False)
    print "conv5", conv5

    conv5_shape = conv5.get_shape().as_list()
    conv5_flat = tf.reshape(conv5, [-1, conv5_shape[1] * conv5_shape[2] * conv5_shape[3]])
    print "conv5_flat", conv5_flat

    layer6 = tf.layers.dense(conv5_flat, 64, activation=tf.nn.relu)

    predictions = tf.layers.dense(layer6, 1, activation=tf.nn.sigmoid)
    print "predictions", predictions
    return predictions

  def create_model(self, model_input, is_training=True, scope="discriminator", **unused_params):
    """Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        is_training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor
    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    print "model_input", model_input

    with tf.variable_scope(scope):
      predictions = self.convnet(model_input, is_training, **unused_params)
    print "predictions", predictions

    return {"predictions": predictions}

