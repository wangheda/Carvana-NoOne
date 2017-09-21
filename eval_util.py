
"""Provides functions to help with evaluating models."""
import datetime
from sklearn.metrics import fbeta_score
import numpy as np

def calculate_f2_score(predictions, actuals):
  """Performs a local (numpy) calculation of the f2_score.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.

  Returns:
    float: The average f2_score across the entire batch.
  """
  predictions = predictions > 0.5
  # fbeta_score throws a confusing error if inputs are not numpy arrays
  predictions, actuals, = np.array(predictions), np.array(actuals)
  # We need to use average='samples' here, any other average method will generate bogus results
  return fbeta_score(actuals, predictions, beta=2, average='samples')

def calculate_f1_score(predictions, actuals):
  """Performs a local (numpy) calculation of the f1_score.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.

  Returns:
    float: The average f2_score across the entire batch.
  """
  predictions = predictions > 0.5
  # fbeta_score throws a confusing error if inputs are not numpy arrays
  predictions, actuals, = np.array(predictions), np.array(actuals)
  # We need to use average='samples' here, any other average method will generate bogus results
  return fbeta_score(actuals, predictions, beta=1, average='samples')

def calculate_hit_at_one(predictions, actuals):
  """Performs a local (numpy) calculation of the hit at one.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.

  Returns:
    float: The average hit at one across the entire batch.
  """
  top_prediction = np.argmax(predictions, 1)
  hits = actuals[np.arange(actuals.shape[0]), top_prediction]
  return np.average(hits)

def calculate_precision_at_equal_recall_rate(predictions, actuals):
  """Performs a local (numpy) calculation of the PERR.

  Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch' x 'num_classes'.

  Returns:
    float: The average precision at equal recall rate across the entire batch.
  """
  aggregated_precision = 0.0
  num_videos = actuals.shape[0]
  for row in np.arange(num_videos):
    num_labels = int(np.sum(actuals[row]))
    top_indices = np.argpartition(predictions[row],
                                     -num_labels)[-num_labels:]
    item_precision = 0.0
    for label_index in top_indices:
      if predictions[row][label_index] > 0:
        item_precision += actuals[row][label_index]
    item_precision /= top_indices.size
    aggregated_precision += item_precision
  aggregated_precision /= num_videos
  return aggregated_precision

class EvaluationMetrics(object):
  """A class to store the evaluation metrics."""

  def __init__(self, num_class):
    """Construct an EvaluationMetrics object to store the evaluation metrics.

    Args:
      num_class: A positive integer specifying the number of classes.
      top_k: A positive integer specifying how many predictions are considered per video.

    Raises:
      ValueError: An error occurred when MeanAveragePrecisionCalculator cannot
        not be constructed.
    """
    self.sum_hit_at_one = 0.0
    self.sum_perr = 0.0
    self.sum_f1score = 0.0
    self.sum_f2score = 0.0
    self.sum_loss = 0.0
    self.num_examples = 0

  def accumulate(self, predictions, labels, loss):
    """Accumulate the metrics calculated locally for this mini-batch.

    Args:
      predictions: A numpy matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
      labels: A numpy matrix containing the ground truth labels.
        Dimensions are 'batch' x 'num_classes'.
      loss: A numpy array containing the loss for each sample.

    Returns:
      dictionary: A dictionary storing the metrics for the mini-batch.

    Raises:
      ValueError: An error occurred when the shape of predictions and actuals
        does not match.
    """
    batch_size = labels.shape[0]
    mean_hit_at_one = calculate_hit_at_one(predictions, labels)
    mean_perr = calculate_precision_at_equal_recall_rate(predictions, labels)
    mean_f1score = calculate_f1score(predictions, labels)
    mean_f2score = calculate_f2score(predictions, labels)
    mean_perr = calculate_precision_at_equal_recall_rate(predictions, labels)
    mean_loss = np.mean(loss)

    self.num_examples += batch_size
    self.sum_hit_at_one += mean_hit_at_one * batch_size
    self.sum_perr += mean_perr * batch_size
    self.sum_f1score += mean_f1score * batch_size
    self.sum_f2score += mean_f2score * batch_size
    self.sum_loss += mean_loss * batch_size

    return {"hit_at_one": mean_hit_at_one, "perr": mean_perr, "f1score": mean_f1score, "f2score": mean_f2score, "loss": mean_loss}

  def get(self):
    """Calculate the evaluation metrics for the whole epoch.

    Raises:
      ValueError: If no examples were accumulated.

    Returns:
      dictionary: a dictionary storing the evaluation metrics for the epoch. The
        dictionary has the fields: avg_hit_at_one, avg_perr, avg_loss, and
        aps (default nan).
    """
    if self.num_examples <= 0:
      raise ValueError("total_sample must be positive.")
    avg_hit_at_one = self.sum_hit_at_one / self.num_examples
    avg_perr = self.sum_perr / self.num_examples
    avg_loss = self.sum_loss / self.num_examples

    epoch_info_dict = {"avg_hit_at_one": avg_hit_at_one, 
            "avg_perr": avg_perr,
            "avg_f1score": avg_f1score,
            "avg_f2score": avg_f2score,
            "avg_loss": avg_loss}
    return epoch_info_dict

  def clear(self):
    """Clear the evaluation metrics and reset the EvaluationMetrics object."""
    self.sum_hit_at_one = 0.0
    self.sum_perr = 0.0
    self.sum_loss = 0.0
    self.map_calculator.clear()
    self.global_ap_calculator.clear()
    self.num_examples = 0


