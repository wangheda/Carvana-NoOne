
import sys
import readers
import tensorflow as tf
from tensorflow import gfile

def get_csv_header():
  return "img,rle_mask"

def convert_id_values_to_csv(image_id, value_list):
  pair_list = []
  value_list = map(int, value_list)
  last_zero = -1
  for i in xrange(len(value_list)):
    if value_list[i] == 0:
      begin_index = last_zero + 1
      num_ones = i - begin_index
      if num_ones > 0:
        pair_list.append(begin_index + 1)
        pair_list.append(num_ones)
      last_zero = i
  return "%s.jpg,%s" % (image_id, " ".join(map(str, pair_list)))

def print_id_array_to_str(image_id, np_array):
  return "%s\t%s" % (image_id, "\t".join(map(str, np_array.flatten("C").tolist())))

def convert_id_array_to_csv(image_id, np_array):
  pair_list = []
  value_list = np_array.flatten("C").tolist()
  last_zero = -1
  for i in xrange(len(value_list)):
    if value_list[i] == 0:
      begin_index = last_zero + 1
      num_ones = i - begin_index
      if num_ones > 0:
        pair_list.append(begin_index + 1)
        pair_list.append(num_ones)
      last_zero = i
  return "%s.jpg,%s" % (image_id, " ".join(map(str, pair_list)))


#############################
#
#        UNIT TESTS
#
#############################

# This function is for the generation of training mask
# to validate the correctness of test utilities
def build_get_mask_graph(data_pattern):
  reader = readers.CarvanaFeatureReader(
              height=1280, 
              width=1918, 
              channels=3)
  files = gfile.Glob(data_pattern)
  print >> sys.stderr, files
  if not files:
    raise IOError("Unable to find training files. data_pattern='" +
                  data_pattern + "'.")
  filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
  training_data = reader.prepare_reader(filename_queue)
  image_id, image_data, image_mask = tf.train.batch(
      training_data,
      batch_size=4,
      capacity=4,
      allow_smaller_final_batch=True,
      enqueue_many=True)
  print >> sys.stderr, image_id
  print >> sys.stderr, image_data
  print >> sys.stderr, image_mask
  return image_id, image_mask


# This function is for the generation of training mask
# to validate the correctness of test utilities
def get_id_array_pairs(data_pattern):
  with tf.device("/cpu:0"):
    image_id, mask = build_get_mask_graph(data_pattern)
    mask = tf.cast(mask, tf.int32)
    init_op = tf.local_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      try:
        while not coord.should_stop():
          id_val, mask_val = sess.run([image_id, mask])
          num_examples = mask_val.shape[0]
          for i in xrange(num_examples):
            print print_id_array_to_str(id_val[i], mask_val[i,:,:])
      except tf.errors.OutOfRangeError:
        print >> sys.stderr, "test_get_id_array_pairs: epoch limit reached."
      finally:
        # When done, ask the threads to stop.
        coord.request_stop()
      # Wait for threads to finish.
      coord.join(threads)


# This function is for the generation of training mask
# to validate the correctness of test utilities
def convert_id_array_txt_to_csv():
  print get_csv_header()
  for line in sys.stdin:
    contents = line.strip().split("\t")
    image_id = contents[0]
    value_list = contents[1:]
    print convert_id_values_to_csv(image_id, value_list)


# This function is for the generation of training mask
# to validate the correctness of test utilities
def total_process(data_pattern):
  print get_csv_header()
  with tf.device("/cpu:0"):
    image_id, mask = build_get_mask_graph(data_pattern)
    mask = tf.cast(mask, tf.int32)
    init_op = tf.local_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      try:
        while not coord.should_stop():
          id_val, mask_val = sess.run([image_id, mask])
          num_examples = mask_val.shape[0]
          for i in xrange(num_examples):
            print convert_id_array_to_csv(id_val[i], mask_val[i,:,:])
      except tf.errors.OutOfRangeError:
        print >> sys.stderr, "test_get_id_array_pairs: epoch limit reached."
      finally:
        # When done, ask the threads to stop.
        coord.request_stop()
      # Wait for threads to finish.
      coord.join(threads)

if __name__ == "__main__":
  helper_msg = "Usage: python %s [test_gen_id_array_pairs|test_convert_id_array_to_csv|test_total_process]" % sys.argv[0]
  if len(sys.argv) >= 2:
    if sys.argv[1] == "test_gen_id_array_pairs":
      get_id_array_pairs("train-data/*.tfrecord")
    elif sys.argv[1] == "test_convert_id_array_to_csv":
      convert_id_array_txt_to_csv()
    elif sys.argv[1] == "test_total_process":
      total_process("train-data/*.tfrecord")
    else:
      sys.exit(helper_msg)
  else:
    sys.exit(helper_msg)


