# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides readers configured for different datasets."""

import sys
import utils
import tensorflow as tf
from tensorflow import logging
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_bool("use_data_augmentation", False,
    "Whether to augmenting images before apply them.")

class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()

def image_augmentation(image, mask):
  """Returns (maybe) augmented images
  (1) Random flip (left <--> right)
  (2) Random flip (up <--> down)
  (3) Random brightness
  (4) Random hue
  Args:
  image (3-D Tensor): Image tensor of (H, W, C)
  mask (3-D Tensor): Mask image tensor of (H, W, 1)
  Returns:
  image: Maybe augmented image (same shape as input `image`)
  mask: Maybe augmented mask (same shape as input `mask`)
  """
  concat_image = tf.concat([image, tf.cast(tf.expand_dims(mask, axis=2), tf.uint8)], axis=-1)

  maybe_flipped = tf.image.random_flip_left_right(concat_image)

  image = maybe_flipped[:, :, :-1]
  mask = tf.cast(maybe_flipped[:, :, -1], tf.bool)

  image = tf.image.random_brightness(image, 0.1)
  image = tf.image.random_hue(image, 0.1)

  return image, mask

class CarvanaFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(self,
               width=1918,
               height=1280,
               channels=3):
    """Construct a CarvanaFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """
    self.width = width
    self.height = height
    self.channels = channels

  def prepare_reader(self, filename_queue, batch_size=16):
    """Creates a single reader thread for .

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)

    feature_map = {"id": tf.FixedLenFeature([], tf.string),
                   "image": tf.FixedLenFeature([], tf.string),
                   "mask": tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(serialized_examples, features=feature_map)
    print >> sys.stderr, " features", features

    image_id = features["id"]
    image_data = features["image"]
    image_mask = features["mask"]
    print >> sys.stderr, " image_id", image_id
    print >> sys.stderr, " image_data", image_data
    print >> sys.stderr, " image_mask", image_mask

    # reshape to rank1
    image_id = tf.reshape(image_id, shape=[1])

    # [height, width, channels]
    image_data = tf.image.decode_jpeg(image_data, channels=3)
    # image_data.set_shape(self.height * self.width * self.channels)
    image_data = tf.reshape(image_data, shape=[self.height, self.width, self.channels])
    print >> sys.stderr, " image_data", image_data

    # [height, width]
    image_mask = tf.decode_raw(image_mask, tf.uint8)
    image_mask.set_shape(self.height * self.width)
    image_mask = tf.reshape(image_mask, shape=[self.height, self.width])
    image_mask = tf.greater(image_mask, 0)
    print >> sys.stderr, " image_mask", image_mask

    # image augmentation
    if hasattr(FLAGS, "use_data_augmentation") and FLAGS.use_data_augmentation:
      image_data, image_mask = image_augmentation(image_data, image_mask)

    image_data = tf.reshape(image_data, shape=[1, self.height, self.width, self.channels])
    image_mask = tf.reshape(image_mask, shape=[1, self.height, self.width])
    return image_id, image_data, image_mask


class CarvanaPredictionFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(self,
               width=1918,
               height=1280,
               channels=3):
    """Construct a CarvanaFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """
    self.width = width
    self.height = height
    self.channels = channels

  def prepare_reader(self, filename_queue, batch_size=16):
    """Creates a single reader thread for .

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)

    feature_map = {"id": tf.FixedLenFeature([], tf.string),
                   "image": tf.FixedLenFeature([], tf.string),
                   "mask": tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(serialized_examples, features=feature_map)
    print >> sys.stderr, " features", features

    image_id = features["id"]
    image_data = features["image"]
    image_mask = features["mask"]
    print >> sys.stderr, " image_id", image_id
    print >> sys.stderr, " image_data", image_data
    print >> sys.stderr, " image_mask", image_mask

    # reshape to rank1
    image_id = tf.reshape(image_id, shape=[1])

    # [height, width, channels]
    image_data = tf.decode_raw(image_data, tf.uint8)
    image_data.set_shape(self.height * self.width * self.channels)
    image_data = tf.reshape(image_data, shape=[self.height, self.width, self.channels])
    print >> sys.stderr, " image_data", image_data

    # [height, width]
    image_mask = tf.decode_raw(image_mask, tf.uint8)
    image_mask.set_shape(self.height * self.width)
    image_mask = tf.reshape(image_mask, shape=[self.height, self.width])
    image_mask = tf.greater(image_mask, 0)
    print >> sys.stderr, " image_mask", image_mask

    # image augmentation
    if hasattr(FLAGS, "use_data_augmentation") and FLAGS.use_data_augmentation:
      image_data, image_mask = image_augmentation(image_data, image_mask)

    image_data = tf.reshape(image_data, shape=[1, self.height, self.width, self.channels])
    image_mask = tf.reshape(image_mask, shape=[1, self.height, self.width])
    return image_id, image_data, image_mask


class CarvanaTestFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(self,
               width=1918,
               height=1280,
               channels=3):
    """Construct a CarvanaFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """
    self.width = width
    self.height = height
    self.channels = channels

  def prepare_reader(self, filename_queue, batch_size=16):
    """Creates a single reader thread for .

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)

    feature_map = {"id": tf.FixedLenFeature([], tf.string),
                   "image": tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(serialized_examples, features=feature_map)
    print >> sys.stderr, " features", features

    image_id = features["id"]
    image_data = features["image"]
    print >> sys.stderr, " image_id", image_id
    print >> sys.stderr, " image_data", image_data

    # reshape to rank1
    image_id = tf.reshape(image_id, shape=[1])

    # [height, width, channels]
    image_data = tf.image.decode_jpeg(image_data, channels=3)
    # image_data.set_shape(self.height * self.width * self.channels)
    image_data = tf.reshape(image_data, shape=[1, self.height, self.width, self.channels])
    print >> sys.stderr, " image_data", image_data

    return image_id, image_data

