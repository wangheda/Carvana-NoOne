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
"""Binary for testing Tensorflow models on the Kaggle Planet dataset."""

import json
import os
import time
import numpy

import test_util
import utils
import image_models
import readers
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "model",
                      "The directory to load the model files from.")
  flags.DEFINE_string("model_checkpoint_path", "",
                      "The file to load the model files from. ")
  flags.DEFINE_string(
      "output_file", "test.out",
      "File that contains the csv predictions")
  flags.DEFINE_string(
      "test_data_list", None,
      "List that contains testing data path")
  flags.DEFINE_string(
      "test_data_pattern", "test-data/*.tfrecord",
      "Pattern for testing data path")
  flags.DEFINE_integer("image_width", 1918, "Width of the image.")
  flags.DEFINE_integer("image_height", 1280, "Height of the image.")
  flags.DEFINE_integer("image_channels", 3, "Channels of the image.")

  flags.DEFINE_string(
      "model", "BasicUNetModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")

  # Model flags.
  flags.DEFINE_integer("batch_size", 4,
                       "How many examples to process per batch for testing.")
  flags.DEFINE_float(
      "prediction_threshold", 0.5,
      "Which value to use as a threshold of true and false values")

  # Other flags.
  flags.DEFINE_boolean("run_once", False, "Whether to run test only once.")


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def get_input_test_tensors(reader,
                                 data_list,
                                 data_pattern,
                                 batch_size=32,
                                 nfold=5,
                                 nfold_index=0):

  logging.info("Using batch size of " + str(batch_size) + " for test.")
  with tf.name_scope("test_input"):
    if data_list is not None:
      with open(data_list) as F:
        files = [line.strip() for line in F.readlines() if len(line.strip()) > 0]
    else:
      files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the test files.")
    logging.info("number of test files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    test_data = reader.prepare_reader(filename_queue)

    return tf.train.batch(
        test_data,
        batch_size=batch_size,
        capacity=3 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)

def build_graph(reader,
                model,
                test_data_list,
                test_data_pattern,
                prediction_threshold=0.5,
                batch_size=8):
  """Creates the Tensorflow graph for test.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    test_data_pattern: glob path to the test data files.
    batch_size: How many examples to process at a time.
    num_readers: How many threads to use for I/O operations.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")
  image_id, image_data = get_input_test_tensors(  # pylint: disable=g-line-too-long
      reader,
      test_data_list,
      test_data_pattern,
      batch_size=batch_size)

  model_input = image_data
  tf.summary.histogram("test/input", model_input)

  feature_dim = len(model_input.get_shape()) - 1
  with tf.name_scope("model"):
    result = model.create_model(model_input,
                                is_training=False)
    for variable in slim.get_model_variables():
      tf.summary.histogram(variable.op.name, variable)

    predictions = tf.cast(result["predictions"], tf.int32)
    tf.summary.histogram("test/predictions", predictions)

    num_examples = tf.shape(predictions)[0]

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("id_batch", image_id)
  tf.add_to_collection("num_examples", num_examples)
  tf.add_to_collection("summary_op", tf.summary.merge_all())


def test_loop(id_batch, prediction_batch, num_examples,
              summary_op, saver, summary_writer, last_global_step_val):
  """Run the test loop once.

  Args:
    id_batch: a tensor of video ids mini-batch.
    prediction_batch: a tensor of predictions mini-batch.
    summary_op: a tensor which runs the tensorboard summary operations.
    saver: a tensorflow saver to restore the model.
    summary_writer: a tensorflow summary_writer
    last_global_step_val: the global step used in the previous test.

  Returns:
    The global_step used in the latest model.
  """

  global_step_val = -1
  with tf.Session() as sess:
    if FLAGS.model_checkpoint_path:
      checkpoint = FLAGS.model_checkpoint_path
    else:
      checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
    if checkpoint:
      logging.info("Loading checkpoint for test: " + checkpoint)
      # Restores from checkpoint
      saver.restore(sess, checkpoint)
      # Assuming model_checkpoint_path looks something like:
      # /my-favorite-path/yt8m_train/model.ckpt-0, extract global_step from it.
      global_step_val = checkpoint.split("/")[-1].split("-")[-1]
    else:
      logging.info("No checkpoint file found.")
      return global_step_val

    if global_step_val == last_global_step_val:
      logging.info("skip this checkpoint global_step_val=%s "
                   "(same as the previous one).", global_step_val)
      return global_step_val

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    fetches = [id_batch, prediction_batch, num_examples, summary_op]
    coord = tf.train.Coordinator()
    
    with open(FLAGS.output_file, "w") as F_test:
      print >> F_test, test_util.get_csv_header()
      try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(
              sess, coord=coord, daemon=True,
              start=True))
        logging.info("enter test loop global_step_val = %s. ",
                     global_step_val)

        examples_processed = 0

        while not coord.should_stop():
          batch_start_time = time.time()

          id_val, predictions_val, num_examples_val, summary_val = sess.run(fetches)

          num = predictions_val.shape[0]
          for i in xrange(num):
            print >> F_test, test_util.convert_id_array_to_csv(id_val[i], predictions_val[i,:,:])

          seconds_per_batch = time.time() - batch_start_time
          example_per_second = num_examples_val / seconds_per_batch

          examples_processed += num_examples_val

          logging.info("examples_processed: %d", examples_processed)

      except tf.errors.OutOfRangeError as e:
        logging.info(
            "Done with batched inference. Now calculating global performance "
            "metrics.")

      except Exception as e:  # pylint: disable=broad-except
        logging.info("Unexpected exception: " + str(e))
        coord.request_stop(e)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)

    return global_step_val


def test():
  tf.set_random_seed(0)  # for reproducibility
  with tf.Graph().as_default():
    # convert feature_names and feature_sizes to lists of values
    reader = readers.CarvanaTestFeatureReader(
                height=FLAGS.image_height, 
                width=FLAGS.image_width, 
                channels=FLAGS.image_channels)

    print FLAGS.model
    print dir(image_models)
    model = find_class_by_name(FLAGS.model, [image_models])()

    if FLAGS.test_data_pattern is "":
      raise IOError("'test_data_pattern' was not specified. " +
                     "Nothing to test.")

    build_graph(
        reader=reader,
        model=model,
        test_data_list=FLAGS.test_data_list,
        test_data_pattern=FLAGS.test_data_pattern,
        prediction_threshold=FLAGS.prediction_threshold,
        batch_size=FLAGS.batch_size)

    logging.info("built test graph")

    id_batch = tf.get_collection("id_batch")[0]
    prediction_batch = tf.get_collection("predictions")[0]
    num_examples = tf.get_collection("num_examples")[0]
    summary_op = tf.get_collection("summary_op")[0]

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir, graph=tf.get_default_graph())

    last_global_step_val = -1
    while True:
      last_global_step_val = test_loop(id_batch, prediction_batch, num_examples,
                                       summary_op, saver, summary_writer,
                                       last_global_step_val)
      if FLAGS.run_once:
        break


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  test()


if __name__ == "__main__":
  app.run()

