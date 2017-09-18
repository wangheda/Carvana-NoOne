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

import utils
import tensorflow as tf
from tensorflow import logging


class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


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
    print " features", features

    image_id = features["id"]
    image_data = features["image"]
    image_mask = features["mask"]
    print " image_id", image_id
    print " image_data", image_data
    print " image_mask", image_mask

    # reshape to rank1
    image_id = tf.reshape(image_id, shape=[1])

    # [height, width, channels]
    image_data = tf.image.decode_jpeg(image_data, channels=3)
    # image_data.set_shape(self.height * self.width * self.channels)
    image_data = tf.reshape(image_data, shape=[1, self.height, self.width, self.channels])
    print " image_data", image_data

    # [height, width]
    image_mask = tf.decode_raw(image_mask, tf.uint8)
    image_mask.set_shape(self.height * self.width)
    image_mask = tf.reshape(image_mask, shape=[1, self.height, self.width])
    image_mask = tf.greater(image_mask, 0)
    print " image_mask", image_mask

    return image_id, image_data, image_mask

