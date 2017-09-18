
import numpy as np
import tensorflow as tf
import os
import sys
import random
from skimage import io
from PIL import Image
from tensorflow import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("jpg_dir", None, "The directory of the image.")
flags.DEFINE_string("gif_dir", None, "The directory of the mask.")
flags.DEFINE_string("output_dir", None, "The directory to put the tfrecords.")
flags.DEFINE_string("placeholder_mask", "test-masks/placeholder.gif", "The placeholder of mask in test dataset.")


def load_jpg(jpg_dir, jpg_file):
  assert jpg_dir is not None, "jpg_dir should not be None"
  jpg_path = os.path.join(jpg_dir, jpg_file)

  assert jpg_path.endswith(".jpg"), "jpg file %s is not in .jpg extension" % jpg_file
  assert os.path.isfile(jpg_path), "jpg file %s does not exist" % jpg_path
  with open(jpg_path, "rb") as F:
    img_str = F.read()

  return img_str


def load_gif(gif_dir, gif_file):
  if gif_dir is not None:
    gif_path = os.path.join(gif_dir, gif_file)
  else:
    return None

  assert gif_path.endswith(".gif"), "gif file %s is not in .gif extension" % gif_file
  assert os.path.isfile(gif_path), "gif file %s does not exist" % gif_path
  # with open(gif_path, "rb") as F:
  #   img_str = F.read()
  img_str = np.array(Image.open(gif_path)).tostring()

  return img_str


def write_to_test_record(id_batch, image_batch, file_id):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_dir + '/' + 'data-%04d.tfrecord' % file_id)
    for i in xrange(len(id_batch)):
        item_id = id_batch[i]
        example = get_output_feature(item_id, [image_batch[i]], ['image'])
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()


def write_to_record(id_batch, image_batch, mask_batch, file_id):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_dir + '/' + 'data-%04d.tfrecord' % file_id)
    for i in xrange(len(id_batch)):
        item_id = id_batch[i]
        example = get_output_feature(item_id, [image_batch[i], mask_batch[i]], ['image', 'mask'])
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_output_feature(item_id, features, names):
    feature_maps = {'id': _bytes_feature(item_id)}
    for i in range(len(names)):
        feature_maps[names[i]] = _bytes_feature(features[i])
    example = tf.train.Example(features=tf.train.Features(feature=feature_maps))
    return example


if __name__=="__main__":
  gif_dir = FLAGS.gif_dir
  jpg_dir = FLAGS.jpg_dir
  
  basenames = [filename[:-4] for filename in os.listdir(jpg_dir) if filename.endswith(".jpg")]

  id_batch = []
  mask_batch = []
  image_batch = []
  file_id = 0

  for basename in basenames:
    # image id
    id_batch.append(basename)
    # jpg image
    jpg_file = basename + ".jpg"
    image = load_jpg(jpg_dir, jpg_file)
    image_batch.append(image)
    # gif image
    gif_file = basename + "_mask.gif"
    if gif_dir is not None:
      mask = load_gif(gif_dir, gif_file)
      mask_batch.append(mask)
    
    if len(id_batch) >= 16:
      print >> sys.stderr, "writing data-%04d.tfrecord ..." % file_id
      if gif_dir is None:
        write_to_test_record(id_batch, image_batch, file_id)
      else:
        write_to_record(id_batch, image_batch, mask_batch, file_id)
      id_batch = []
      image_batch = []
      mask_batch = []
      file_id += 1

  if len(id_batch) > 0:
    print >> sys.stderr, "writing data-%04d.tfrecord ..." % file_id
    if gif_dir is None:
      write_to_test_record(id_batch, image_batch, file_id)
    else:
      write_to_record(id_batch, image_batch, mask_batch, file_id)
    id_batch = []
    image_batch = []
    mask_batch = []
    file_id += 1

