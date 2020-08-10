import os 
from typing import List

import tensorflow as tf 

PREPROCESSING_STRATEGIES = ["NO", "CENTER_CROP_THEN_SCALE", "SQUARE_MARGIN_THEN_SCALE"]
IMAGE_EXTENSIONS = ["jpeg", "jpg", "png"]

# Helper functions

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class TfRecordManager:
    def __init__(self, dir: str, prefix: str, size=5000):
        """This class manage the creation of TFRecord files

        Args:
            size: Number of images to write in one tfrecord file
            dir: path to the dir where tfrecord files will be written
            prefix: prefix of the tfrecord. The filenames will be as "prefix_n.tfrecord" with n starting from 0 and incrementing at each new file
        """
        self.size = size
        self.dir = dir
        self.prefix = prefix
        self.n_images = 0  # number of images in the tfrecord
        self.index = 0  # index of the tfrecord
        self.writer = None

        # prepare the directory
        os.makedirs(self.dir, exist_ok=True)

    def add(self, example):
        """ Add an tf example to the tfrecord. 

        Args:
            example: The example to save in the tfrecord
        """
        if self.writer is None:
            filename = os.path.join(self.dir, f'{self.prefix}_{self.index}.tfrecord')
            self.writer = tf.io.TFRecordWriter(filename)

        self.writer.write(example.SerializeToString())
        self.n_images += 1

        # Check if need to close the writer
        if self.n_images == self.size:
            self.writer.close()
            self.n_images = 0
            self.index += 1
            self.writer = None

class Split:
    def __init__(self, name, images_dir_path, annotation_file_path=None, delimiter=',', header_exists=False):
        """
        Args:
            name: dataset split name either train/val/test
            images_dir_path: directory path for the images
            annotation_file_path: annotation file path in the format `1st column`: images names, `2nd column`: label.
                if it is `None` it will be assumed that images are stored in the sub directories of `annotation_file_path`
                by the name of their classes (labels)
            delimiter: separator for the columns annotation file, this is irrelevant if `annotation_file_path` is `None`
            header_exists: whether annotation file has header or not, this is irrelevant if `annotation_file_path` is `None`
        """
        self.name = name
        self.images_dir_path = images_dir_path
        self.annotation_file_path = annotation_file_path
        self.delimiter = delimiter
        self.header_exists = header_exists
