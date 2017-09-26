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
"""Binary for training Tensorflow models on the YouTube-8M dataset."""

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
                      "The directory to save the model files in.")
  flags.DEFINE_string(
      "train_data_pattern", "train-predictions/*.tfrecord",
      "Pattern for training data path for refiner network")
  flags.DEFINE_string(
      "true_label_pattern", "train-data/*.tfrecord",
      "Pattern for true labels")

  flags.DEFINE_integer("image_width", 1918, "Width of the image.")
  flags.DEFINE_integer("image_height", 1280, "Height of the image.")
  flags.DEFINE_integer("image_channels", 1, "Channels of the image.")

  flags.DEFINE_string(
      "refiner_model", "RefinerScaledUNetModel",
      "Which architecture to use for the refiner model. Models are defined "
      "in models.py.")
  flags.DEFINE_string(
      "discriminator_model", "DiscriminatorConvNetModel",
      "Which architecture to use for the discriminator model. Models are defined "
      "in models.py.")
  flags.DEFINE_bool(
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")

  # Training flags.
  flags.DEFINE_integer("batch_size", 4,
                       "How many examples to process per batch for training.")
  flags.DEFINE_string("refiner_loss", "CrossEntropyLoss",
                      "Which loss function to use for the refiner.")
  flags.DEFINE_string("similarity_loss", "MeanSquareErrorLoss",
                      "Which loss function to use for regulating the diff between"
                      " the prediction and the input.")
  flags.DEFINE_string("discriminator_loss", "CrossEntropyLoss",
                      "Which loss function to use for the discriminator.")
  flags.DEFINE_float(
      "l2_penalty", 1e-7,
      "The penalty given to regulization")
  flags.DEFINE_float(
      "regularization_penalty", 1,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")
  flags.DEFINE_float(
      "prediction_threshold", 0.5,
      "Which value to use as a threshold of true and false values")
  flags.DEFINE_float("base_learning_rate", 0.001,
                     "Which learning rate to start with.")
  flags.DEFINE_float("learning_rate_decay", 0.99,
                     "Learning rate decay factor to be applied every "
                     "learning_rate_decay_examples.")
  flags.DEFINE_float("learning_rate_decay_examples", 4000,
                     "Multiply current learning rate by learning_rate_decay "
                     "every learning_rate_decay_examples.")
  flags.DEFINE_integer("num_epochs", 10,
                       "How many passes to make over the dataset before "
                       "halting training.")
  flags.DEFINE_integer("max_steps", None,
                       "How many steps before stop.")
  flags.DEFINE_float("keep_checkpoint_every_n_hours", 0.01,
                     "How many hours before saving a new checkpoint")
  flags.DEFINE_integer("keep_checkpoint_interval", 1,
                     "How many minutes before saving a new checkpoint")

  # Other flags.
  flags.DEFINE_integer("num_readers", 4,
                       "How many threads to use for reading input files.")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
  flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
  flags.DEFINE_bool(
      "log_device_placement", False,
      "Whether to write the device on which every op will run into the "
      "logs on startup.")

  flags.DEFINE_bool("accumulate_gradients", False,
      "Whether to accumulate gradients of several batches before apply them.")
  flags.DEFINE_integer("apply_every_n_batches", 2,
      "How many batches of gradients to compute before apply them together.")


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def validate_class_name(flag_value, category, modules, expected_superclass):
  """Checks that the given string matches a class of the expected type.

  Args:
    flag_value: A string naming the class to instantiate.
    category: A string used further describe the class in error messages
              (e.g. 'model', 'reader', 'loss').
    modules: A list of modules to search for the given class.
    expected_superclass: A class that the given class should inherit from.

  Raises:
    FlagsError: If the given class could not be found or if the first class
    found with that name doesn't inherit from the expected superclass.

  Returns:
    True if a class was found that matches the given constraints.
  """
  candidates = [getattr(module, flag_value, None) for module in modules]
  for candidate in candidates:
    if not candidate:
      continue
    if not issubclass(candidate, expected_superclass):
      raise flags.FlagsError("%s '%s' doesn't inherit from %s." %
                             (category, flag_value,
                              expected_superclass.__name__))
    return True
  raise flags.FlagsError("Unable to find %s '%s'." % (category, flag_value))

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=16,
                           num_epochs=None,
                           num_readers=1):
  """Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None'
                to run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    print "number of training files:", len(files)
    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    logging.info("Number of training files: %s.", str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=FLAGS.batch_size * 8,
        min_after_dequeue=FLAGS.batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def optimizing(optimizer, final_loss, clip_gradient_norm, global_step, prefix, scope):
  # Accumulate several batches before gradient descent options
  # to make larger batch than the memory could be able to hold

  tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
  # tvs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  print "tvs", tvs

  if FLAGS.accumulate_gradients:
    assert FLAGS.apply_every_n_batches > 0, "apply_every_n_batches should be > 0"
    scale = 1.0 / FLAGS.apply_every_n_batches

    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs] 
    init_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    gvs = optimizer.compute_gradients(final_loss, tvs)
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

    if clip_gradient_norm > 0:
      with tf.name_scope('clip_grads'):
        clipped_accum_vars = utils.clip_variable_norms(accum_vars, 
                max_norm = clip_gradient_norm, scale = scale)
        apply_op = optimizer.apply_gradients([(clipped_accum_vars[i], gv[1]) 
                for i, gv in enumerate(gvs)], global_step=global_step)
        
    else:
        apply_op = optimizer.apply_gradients([(accum_vars[i] * scale, gv[1]) 
                for i, gv in enumerate(gvs)], global_step=global_step)
    tf.get_collection_ref(prefix + "_train/init_ops").extend(init_ops)
    tf.get_collection_ref(prefix + "_train/accum_ops").extend(accum_ops)
    tf.add_to_collection(prefix + "_train/apply_op", apply_op)

  # the original way, apply every batch
  else:
    gradients = optimizer.compute_gradients(final_loss,
        colocate_gradients_with_ops=False, var_list=tvs)
    print gradients
    if clip_gradient_norm > 0:
      with tf.name_scope('clip_grads'):
        gradients = utils.clip_gradient_norms(gradients, clip_gradient_norm)
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)
    tf.add_to_collection(prefix + "_train/train_op", train_op)
  return None


def build_graph(refiner_model,
                discriminator_model,
                train_data_reader,
                train_data_pattern,
                true_label_reader,
                true_label_pattern,
                refiner_loss_fn=losses.CrossEntropyLoss(),
                similarity_loss_fn=losses.MeanSquareErrorLoss(),
                discriminator_loss_fn=losses.CrossEntropyLoss(),
                batch_size=4,
                base_learning_rate=0.01,
                learning_rate_decay_examples=4000,
                learning_rate_decay=0.99,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                prediction_threshold=0.5,
                regularization_penalty=1,
                num_readers=2,
                num_epochs=None):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    train_data_pattern: glob path to the training data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
  """
  
  global_step = tf.Variable(0, trainable=False, name="global_step")
  
  if FLAGS.accumulate_gradients:
    actual_batch_size = batch_size * FLAGS.apply_every_n_batches 
  else:
    actual_batch_size = batch_size

  learning_rate = tf.train.exponential_decay(
      base_learning_rate,
      global_step * actual_batch_size,
      learning_rate_decay_examples,
      learning_rate_decay,
      staircase=True)
  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = optimizer_class(learning_rate)

  image_id, image_data, image_mask = (
      get_input_data_tensors(
          train_data_reader,
          train_data_pattern,
          batch_size=batch_size,
          num_readers=num_readers,
          num_epochs=num_epochs))

  _, _, true_mask = (
      get_input_data_tensors(
          true_label_reader,
          true_label_pattern,
          batch_size=batch_size,
          num_readers=num_readers,
          num_epochs=num_epochs))

  model_input = image_data
  tf.summary.histogram("model/input", model_input)

  with tf.name_scope("refiner_model"):
    result = refiner_model.create_model(
        model_input,
        scope="refiner_model",
        l2_penalty=FLAGS.l2_penalty)
    print "result", result

  for variable in slim.get_model_variables():
    tf.summary.histogram(variable.op.name, variable)

  predictions = result["predictions"]
  if "loss" in result.keys():
    label_loss = result["loss"]
  else:
    refiner_loss = refiner_loss_fn.calculate_loss(predictions, image_mask)
    similarity_loss = similarity_loss_fn.calculate_loss(predictions, image_mask)
    tf.summary.scalar("refiner_loss", refiner_loss)
    tf.summary.scalar("similarity_loss", similarity_loss)
    label_loss = refiner_loss + similarity_loss
    tf.summary.scalar("label_loss", label_loss)

    tf.add_to_collection("refiner_loss", refiner_loss)
    tf.add_to_collection("similarity_loss", similarity_loss)

  tf.summary.histogram("model/predictions", predictions)
  tf.summary.scalar("label_loss", label_loss)

  if "regularization_loss" in result.keys():
    reg_loss = result["regularization_loss"]
  else:
    reg_loss = tf.constant(0.0)
  
  reg_losses = tf.losses.get_regularization_losses(scope="refiner_model")
  if reg_losses:
    reg_loss += tf.add_n(reg_losses)
  
  if regularization_penalty != 0:
    tf.summary.scalar("reg_loss", reg_loss)

  # Adds update_ops (e.g., moving average updates in batch normalization) as
  # a dependency to the train_op.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="refiner_model")
  if "update_ops" in result.keys():
    update_ops += result["update_ops"]
  if update_ops:
    with tf.control_dependencies(update_ops):
      barrier = tf.no_op(name="gradient_barrier")
      with tf.control_dependencies([barrier]):
        label_loss = tf.identity(label_loss)

  # Incorporate the L2 weight penalties etc.
  final_loss = regularization_penalty * reg_loss + label_loss
  optimizing(optimizer, final_loss, clip_gradient_norm, global_step, prefix="refiner", scope="refiner_model")

  labels = tf.cast(image_mask, tf.int32)
  float_labels = tf.cast(image_mask, tf.float32)

  bool_predictions = tf.greater(predictions, prediction_threshold)
  true_pos = tf.cast(tf.reduce_sum(tf.cast(labels > 0, tf.int32) * tf.cast(predictions > prediction_threshold, tf.int32)), tf.float32)
  false_pos = tf.cast(tf.reduce_sum(tf.cast(labels <= 0, tf.int32) * tf.cast(predictions > prediction_threshold, tf.int32)), tf.float32)
  false_neg = tf.cast(tf.reduce_sum(tf.cast(labels > 0, tf.int32) * tf.cast(predictions <= prediction_threshold, tf.int32)), tf.float32)
  mean_iou = (2.0 * true_pos + 1e-7) / (2.0 * true_pos + false_pos + false_neg + 1e-7)
  print mean_iou

  num_examples = tf.shape(labels)[0]

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("id_batch", image_id)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("model_input", model_input)
  tf.add_to_collection("num_examples", num_examples)
  tf.add_to_collection("labels", labels)
  tf.add_to_collection("float_labels", float_labels)
  tf.add_to_collection("bool_predictions", bool_predictions)
  tf.add_to_collection("mean_iou", mean_iou)


  def split_into_small_patches(masks, label_value):
    masks = tf.expand_dims(masks, axis=3)
    masks = tf.pad(masks, paddings=[[0,0], [0,0], [1,1], [0,0]])
    PATCH_SIZE = [1, 320, 320, 1]
    HALF_PATCH_SIZE = [1, 160, 160, 1]
    patches = tf.extract_image_patches(masks, PATCH_SIZE, HALF_PATCH_SIZE, [1,1,1,1], "VALID")
    patches = tf.reshape(patches, [-1, 320, 320, 1])
    if label_value == 0:
      labels = tf.zeros([tf.shape(patches)[0],1])
    else:
      labels = tf.ones([tf.shape(patches)[0],1])
    return patches, labels


  with tf.name_scope("discriminator_model"):
    p_patches, p_labels = split_into_small_patches(tf.cast(bool_predictions, tf.float32), 0)
    t_patches, t_labels = split_into_small_patches(tf.cast(true_mask, tf.float32), 1)
    disc_batch = tf.concat([p_patches, t_patches], axis=0)
    disc_labels = tf.concat([p_labels, t_labels], axis=0)
    print "disc_batch", disc_batch
    print "disc_labels", disc_labels

    disc_result = discriminator_model.create_model(
        disc_batch,
        scope="discriminator_model",
        l2_penalty=FLAGS.l2_penalty)

    print "disc_result", disc_result

  for variable in slim.get_model_variables():
    tf.summary.histogram(variable.op.name, variable)

  disc_predictions = disc_result["predictions"]
  if "loss" in disc_result.keys():
    disc_label_loss = disc_result["loss"]
  else:
    disc_label_loss = discriminator_loss_fn.calculate_loss(disc_predictions, disc_labels) * 4000
    tf.summary.scalar("discriminator_loss", disc_label_loss)

  tf.summary.histogram("model/disc_predictions", disc_predictions)

  if "regularization_loss" in disc_result.keys():
    disc_reg_loss = result["regularization_loss"]
  else:
    disc_reg_loss = tf.constant(0.0)
  
  disc_reg_losses = tf.losses.get_regularization_losses(scope="discriminator_model")
  if disc_reg_losses:
    disc_reg_loss += tf.add_n(disc_reg_losses)
  
  if regularization_penalty != 0:
    tf.summary.scalar("disc_reg_loss", disc_reg_loss)

  # Adds update_ops (e.g., moving average updates in batch normalization) as
  # a dependency to the train_op.
  disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="discriminator_model")
  if "update_ops" in disc_result.keys():
    disc_update_ops += disc_result["update_ops"]
  if disc_update_ops:
    with tf.control_dependencies(disc_update_ops):
      disc_barrier = tf.no_op(name="disc_gradient_barrier")
      with tf.control_dependencies([disc_barrier]):
        disc_label_loss = tf.identity(disc_label_loss)

  # Incorporate the L2 weight penalties etc.
  disc_final_loss = regularization_penalty * disc_reg_loss + disc_label_loss
  tf.add_to_collection("discriminator_loss", disc_label_loss)

  optimizing(optimizer, disc_final_loss, clip_gradient_norm, global_step, prefix="discriminator", scope="discriminator_model")

  refiner2_final_loss = regularization_penalty * reg_loss + label_loss - disc_label_loss
  optimizing(optimizer, refiner2_final_loss, clip_gradient_norm, global_step, prefix="refiner2", scope="refiner_model")



class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task, train_dir, log_device_placement=True):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(log_device_placement=log_device_placement)

    if self.is_master and self.task.index > 0:
      raise StandardError("%s: Only one replica of master expected",
                          task_as_string(self.task))

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    if self.is_master and start_new_model:
      self.remove_training_directory(self.train_dir)

    target, device_fn = self.start_server_if_distributed()

    meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

    with tf.Graph().as_default() as graph:

      if meta_filename:
        saver = self.recover_model(meta_filename)

      with tf.device(device_fn):

        if not meta_filename:
          saver = self.build_model()

        global_step = tf.get_collection("global_step")[0]

        loss = tf.get_collection("loss")[0]
        refiner_loss = tf.get_collection("refiner_loss")[0]
        discriminator_loss = tf.get_collection("discriminator_loss")[0]
        similarity_loss = tf.get_collection("similarity_loss")[0]

        predictions = tf.get_collection("predictions")[0]
        labels = tf.get_collection("labels")[0]
        float_labels = tf.get_collection("float_labels")[0]
        num_examples = tf.get_collection("num_examples")[0]

        if FLAGS.accumulate_gradients:
          refiner_init_ops = tf.get_collection("refiner_train/init_ops")
          refiner_accum_ops = tf.get_collection("refiner_train/accum_ops")
          refiner_apply_op = tf.get_collection("refiner_train/apply_op")[0]
          discriminator_init_ops = tf.get_collection("discriminator_train/init_ops")
          discriminator_accum_ops = tf.get_collection("discriminator_train/accum_ops")
          discriminator_apply_op = tf.get_collection("discriminator_train/apply_op")[0]
          refiner2_init_ops = tf.get_collection("refiner2_train/init_ops")
          refiner2_accum_ops = tf.get_collection("refiner2_train/accum_ops")
          refiner2_apply_op = tf.get_collection("refiner2_train/apply_op")[0]
        else:
          refiner_train_op = tf.get_collection("refiner_train/train_op")[0]
          discriminator_train_op = tf.get_collection("discriminator_train/train_op")[0]
          refiner2_train_op = tf.get_collection("refiner2_train/train_op")[0]

        mean_iou = tf.get_collection("mean_iou")[0]
        init_op = tf.global_variables_initializer()


    sv = tf.train.Supervisor(
        graph,
        logdir=self.train_dir,
        init_op=init_op,
        is_chief=self.is_master,
        global_step=global_step,
        save_model_secs=FLAGS.keep_checkpoint_interval * 60,
        save_summaries_secs=120,
        saver=saver)

    mean = lambda x: sum(x) / len(x)

    logging.info("%s: Starting managed session.", task_as_string(self.task))
    with sv.managed_session(target, config=self.config) as sess:

      steps = sess.run(global_step)
      try:
        logging.info("%s: Entering training loop.", task_as_string(self.task))
        while not sv.should_stop():

          steps += 1
          batch_start_time = time.time()

          num_examples_processed = 0

          refiner_stage = 2000
          discriminator_stage = 2000
          interleave_stage = 200

          training_flag = ""

          if steps < refiner_stage:
            training_flag = " refiner_init"
            sub_loss = refiner_loss
            if FLAGS.accumulate_gradients:
              init_ops = refiner_init_ops
              accum_ops = refiner_accum_ops
              apply_op = refiner_apply_op
            else:
              train_op = refiner_train_op
          elif refiner_stage <= steps < refiner_stage + discriminator_stage:
            training_flag = " discriminator_init"
            sub_loss = discriminator_loss
            if FLAGS.accumulate_gradients:
              init_ops = discriminator_init_ops
              accum_ops = discriminator_accum_ops
              apply_op = discriminator_apply_op
            else:
              train_op = discriminator_train_op
          else:
            if ((steps - refiner_stage - discriminator_stage) / interleave_stage) % 2 == 0:
              training_flag = " refiner"
              sub_loss = refiner_loss
              if FLAGS.accumulate_gradients:
                init_ops = refiner2_init_ops
                accum_ops = refiner2_accum_ops
                apply_op = refiner2_apply_op
              else:
                train_op = refiner2_train_op
            else:
              training_flag = " discriminator"
              sub_loss = discriminator_loss
              if FLAGS.accumulate_gradients:
                init_ops = discriminator_init_ops
                accum_ops = discriminator_accum_ops
                apply_op = discriminator_apply_op
              else:
                train_op = discriminator_train_op
              

          if FLAGS.accumulate_gradients:
            # init the buffer to zero
            sess.run(init_ops)
            # compute gradients
            loss_val, sub_loss_val, mean_iou_val = [], [], []
            for i in xrange(FLAGS.apply_every_n_batches):
              ret_list = sess.run([num_examples, loss, sub_loss, mean_iou] + accum_ops)
              num_examples_processed += ret_list[0]
              loss_val.append(ret_list[1])
              sub_loss_val.append(ret_list[2])
              mean_iou_val.append(ret_list[3])
            # accumulate all
            loss_val, sub_loss_val, mean_iou_val = map(mean, [loss_val, sub_loss_val, mean_iou_val])
            _, global_step_val = sess.run([apply_op, global_step])

          else:
            # the original apply-every-batch scheme
            _, global_step_val, loss_val, sub_loss_val, predictions_val, labels_val, mean_iou_val, num_examples_val = sess.run(
                [train_op, global_step, loss, sub_loss, predictions, labels, mean_iou, num_examples])
            num_examples_processed += num_examples_val

          seconds_per_batch = time.time() - batch_start_time

          if self.is_master:
            examples_per_second = num_examples_processed / seconds_per_batch

            logging.info("%s: training step " + str(global_step_val) + 
                         "| IOU: " + ("%.5f" % mean_iou_val) + 
                         " Loss: " + str(loss_val) +
                         " SubLoss: " + str(sub_loss_val) +
                         " " + training_flag, task_as_string(self.task))

            sv.summary_writer.add_summary(
                utils.MakeSummary(
                    "model/Training_IOU", mean_iou_val), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary(
                    "global_step/Examples/Second", examples_per_second), global_step_val)
            sv.summary_writer.flush()

          if FLAGS.max_steps is not None and steps > FLAGS.max_steps:
            logging.info("%s: Done training -- max_steps limit reached.",
                         task_as_string(self.task))
            break

      except tf.errors.OutOfRangeError:
        logging.info("%s: Done training -- epoch limit reached.",
                     task_as_string(self.task))

    logging.info("%s: Exited training loop.", task_as_string(self.task))
    sv.Stop()

  def build_model(self):
    """Find the model and build the graph."""

    # reader
    train_data_reader = readers.CarvanaPredictionFeatureReader(
                height=FLAGS.image_height, 
                width=FLAGS.image_width, 
                channels=FLAGS.image_channels)

    true_label_reader = readers.CarvanaFeatureReader(
                height=FLAGS.image_height, 
                width=FLAGS.image_width, 
                channels=3)

    # Find the model.
    refiner_model = find_class_by_name(FLAGS.refiner_model, [image_models])()
    discriminator_model = find_class_by_name(FLAGS.discriminator_model, [image_models])()
    # label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

    # build the graph
    build_graph(refiner_model=refiner_model,
                discriminator_model=discriminator_model,
                train_data_reader=train_data_reader,
                train_data_pattern=FLAGS.train_data_pattern,
                true_label_reader=true_label_reader,
                true_label_pattern=FLAGS.true_label_pattern,
                optimizer_class=optimizer_class,
                clip_gradient_norm=FLAGS.clip_gradient_norm,
                base_learning_rate=FLAGS.base_learning_rate,
                learning_rate_decay=FLAGS.learning_rate_decay,
                learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                prediction_threshold=FLAGS.prediction_threshold,
                regularization_penalty=FLAGS.regularization_penalty,
                num_readers=FLAGS.num_readers,
                batch_size=FLAGS.batch_size,
                num_epochs=FLAGS.num_epochs)

    logging.info("%s: Built graph.", task_as_string(self.task))

    return tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)

  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""

    if self.cluster:
      logging.info("%s: Starting trainer within cluster %s.",
                   task_as_string(self.task), self.cluster.as_dict())
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info(
          "%s: Removing existing train directory.",
          task_as_string(self.task))
      gfile.DeleteRecursively(train_dir)
    except:
      logging.error(
          "%s: Failed to delete directory " + train_dir +
          " when starting a new model. Please delete it manually and" +
          " try again.", task_as_string(self.task))

  def get_meta_filename(self, start_new_model, train_dir):
    if start_new_model:
      logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                   task_as_string(self.task))
      return None
    
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint: 
      logging.info("%s: No checkpoint file found. Building a new model.",
                   task_as_string(self.task))
      return None
    
    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("%s: No meta graph file found. Building a new model.",
                     task_as_string(self.task))
      return None
    else:
      return meta_filename

  def recover_model(self, meta_filename):
    logging.info("%s: Restoring from meta graph file %s",
                 task_as_string(self.task), meta_filename)
    return tf.train.import_meta_graph(meta_filename)


class ParameterServer(object):
  """A parameter server to serve variables in a distributed execution."""

  def __init__(self, cluster, task):
    """Creates a ParameterServer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task

  def run(self):
    """Starts the parameter server."""

    logging.info("%s: Starting parameter server within cluster %s.",
                 task_as_string(self.task), self.cluster.as_dict())
    server = start_server(self.cluster, self.task)
    server.join()


def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
  """

  if not task.type:
    raise ValueError("%s: The task type must be specified." %
                     task_as_string(task))
  if task.index is None:
    raise ValueError("%s: The task index must be specified." %
                     task_as_string(task))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)

def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)

def main(unused_argv):
  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Logging the version.
  logging.set_verbosity(tf.logging.INFO)
  logging.info("%s: Tensorflow version: %s.",
               task_as_string(task), tf.__version__)

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    Trainer(cluster, task, FLAGS.train_dir, FLAGS.log_device_placement).run(
        start_new_model=FLAGS.start_new_model)
  elif task.type == "ps":
    ParameterServer(cluster, task).run()
  else:
    raise ValueError("%s: Invalid task_type: %s." %
                     (task_as_string(task), task.type))


if __name__ == "__main__":
  app.run()
