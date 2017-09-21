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
"""Binary for evaluating Tensorflow models on the Kaggle Planet dataset."""

import json
import os
import time
import numpy

import eval_util
import utils
import losses
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
      "eval_data_list", None,
      "List that contains evaling data path")
  flags.DEFINE_string(
      "eval_data_pattern", "eval-data/data*.tfrecord",
      "Pattern for evaling data path")
  flags.DEFINE_integer("image_width", 1918, "Width of the image.")
  flags.DEFINE_integer("image_height", 1280, "Height of the image.")
  flags.DEFINE_integer("image_channels", 3, "Channels of the image.")

  flags.DEFINE_string(
      "model", "BasicUNetModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")

  # Model flags.
  flags.DEFINE_integer("batch_size", 8,
                       "How many examples to process per batch for evaluating.")
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Loss computed on validation data")
  flags.DEFINE_float(
      "prediction_threshold", 0.5,
      "Which value to use as a threshold of true and false values")

  # Other flags.
  flags.DEFINE_boolean("run_once", False, "Whether to run eval only once.")


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def get_input_evaluation_tensors(reader,
                                 data_list,
                                 data_pattern,
                                 batch_size=32,
                                 nfold=5,
                                 nfold_index=0):

  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    if data_list is not None:
      with open(data_list) as F:
        files = [line.strip() for line in F.readlines() if len(line.strip()) > 0]
    else:
      files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = reader.prepare_reader(filename_queue)

    return tf.train.batch(
        eval_data,
        batch_size=batch_size,
        capacity=3 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)

def build_graph(reader,
                model,
                eval_data_list,
                eval_data_pattern,
                label_loss_fn,
                prediction_threshold=0.5,
                batch_size=8):
  """Creates the Tensorflow graph for evaluation.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    eval_data_pattern: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    num_readers: How many threads to use for I/O operations.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")
  image_id, image_data,  image_mask = get_input_evaluation_tensors(  # pylint: disable=g-line-too-long
      reader,
      eval_data_list,
      eval_data_pattern,
      batch_size=batch_size)

  model_input = image_data
  tf.summary.histogram("model/input", model_input)

  feature_dim = len(model_input.get_shape()) - 1
  with tf.name_scope("model"):
    result = model.create_model(model_input,
                                is_training=False)
    for variable in slim.get_model_variables():
      tf.summary.histogram(variable.op.name, variable)

    predictions = result["predictions"]
    if "loss" in result.keys():
      label_loss = result["loss"]
    else:
      label_loss = label_loss_fn.calculate_loss(predictions, image_mask)

    tf.summary.histogram("model/predictions", predictions)
    tf.summary.scalar("label_loss", label_loss)

    labels = tf.cast(image_mask, tf.int32)
    float_labels = tf.cast(image_mask, tf.float32)

    auc, _ = tf.metrics.auc(labels, predictions, num_thresholds=40)

    bool_predictions = tf.greater(predictions, prediction_threshold)
    true_pos = tf.cast(tf.reduce_sum(tf.cast(labels > 0, tf.int32) * tf.cast(predictions > prediction_threshold, tf.int32)), tf.float32)
    false_pos = tf.cast(tf.reduce_sum(tf.cast(labels <= 0, tf.int32) * tf.cast(predictions > prediction_threshold, tf.int32)), tf.float32)
    false_neg = tf.cast(tf.reduce_sum(tf.cast(labels > 0, tf.int32) * tf.cast(predictions <= prediction_threshold, tf.int32)), tf.float32)
    mean_iou = (2.0 * true_pos + 1e-7) / (2 * true_pos + false_pos + false_neg + 1e-7)
    print mean_iou

    num_examples = tf.shape(labels)[0]

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("id_batch", image_id)
  tf.add_to_collection("summary_op", tf.summary.merge_all())

  tf.add_to_collection("num_examples", num_examples)
  tf.add_to_collection("labels", labels)
  tf.add_to_collection("float_labels", float_labels)
  tf.add_to_collection("bool_predictions", bool_predictions)
  tf.add_to_collection("auc", auc)
  tf.add_to_collection("mean_iou", mean_iou)


def evaluation_loop(id_batch, prediction_batch, label_batch, loss, mean_iou, num_examples,
                    summary_op, saver, summary_writer, last_global_step_val):
  """Run the evaluation loop once.

  Args:
    id_batch: a tensor of video ids mini-batch.
    prediction_batch: a tensor of predictions mini-batch.
    label_batch: a tensor of label_batch mini-batch.
    loss: a tensor of loss for the examples in the mini-batch.
    summary_op: a tensor which runs the tensorboard summary operations.
    saver: a tensorflow saver to restore the model.
    summary_writer: a tensorflow summary_writer
    evl_metrics: an EvaluationMetrics object.
    last_global_step_val: the global step used in the previous evaluation.

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
      logging.info("Loading checkpoint for eval: " + checkpoint)
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
    fetches = [id_batch, prediction_batch, label_batch, loss, mean_iou, num_examples, summary_op]
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      logging.info("enter eval_once loop global_step_val = %s. ",
                   global_step_val)

      examples_processed = 0
      total_iou_val = 0.0
      total_loss_val = 0.0

      while not coord.should_stop():
        batch_start_time = time.time()

        _, predictions_val, labels_val, loss_val, mean_iou_val, num_examples_val, summary_val = sess.run(fetches)

        seconds_per_batch = time.time() - batch_start_time
        example_per_second = num_examples_val / seconds_per_batch

        examples_processed += num_examples_val
        total_iou_val += mean_iou_val * num_examples_val
        total_loss_val += loss_val * num_examples_val

        logging.info("examples_processed: %d | mean_iou: %.5f", examples_processed, mean_iou_val)

    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")
      # calculate the metrics for the entire epoch
      epoch_info_dict = {}
      epoch_info_dict["epoch_id"] = global_step_val
      epoch_info_dict["mean_iou"] = total_iou_val / examples_processed
      epoch_info_dict["avg_loss"] = total_loss_val / examples_processed

      summary_writer.add_summary(summary_val, global_step_val)
      epochinfo = utils.AddEpochSummary(
          summary_writer,
          epoch_info_dict,
          summary_scope="Eval")
      logging.info(epochinfo)

    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return global_step_val


def evaluate():
  tf.set_random_seed(0)  # for reproducibility
  with tf.Graph().as_default():
    # convert feature_names and feature_sizes to lists of values
    reader = readers.CarvanaFeatureReader(
                height=FLAGS.image_height, 
                width=FLAGS.image_width, 
                channels=FLAGS.image_channels)

    print FLAGS.model
    print dir(image_models)
    model = find_class_by_name(FLAGS.model, [image_models])()
    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()

    if FLAGS.eval_data_pattern is "":
      raise IOError("'eval_data_pattern' was not specified. " +
                     "Nothing to evaluate.")

    build_graph(
        reader=reader,
        model=model,
        eval_data_list=FLAGS.eval_data_list,
        eval_data_pattern=FLAGS.eval_data_pattern,
        label_loss_fn=label_loss_fn,
        prediction_threshold=FLAGS.prediction_threshold,
        batch_size=FLAGS.batch_size)

    logging.info("built evaluation graph")

    id_batch = tf.get_collection("id_batch")[0]
    prediction_batch = tf.get_collection("predictions")[0]
    label_batch = tf.get_collection("labels")[0]
    loss = tf.get_collection("loss")[0]
    mean_iou = tf.get_collection("mean_iou")[0]
    num_examples = tf.get_collection("num_examples")[0]
    summary_op = tf.get_collection("summary_op")[0]

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir, graph=tf.get_default_graph())

    last_global_step_val = -1
    while True:
      last_global_step_val = evaluation_loop(id_batch, prediction_batch,
                                             label_batch, loss, mean_iou, num_examples,
                                             summary_op, saver, summary_writer,
                                             last_global_step_val)
      if FLAGS.run_once:
        break


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  evaluate()


if __name__ == "__main__":
  app.run()

