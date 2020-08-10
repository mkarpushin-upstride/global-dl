import os
import glob
import logging
import random 
import yaml
import sys
from typing import List

from cv2 import cv2
import tensorflow as tf 

from create_tf_utils import (bytes_feature, bytes_list_feature, float_list_feature, int64_feature, int64_list_feature, 
                            TfRecordManager, Split, PREPROCESSING_STRATEGIES, IMAGE_EXTENSIONS)

from argument_parser import parse_cmd

def load_and_preprocess_img(img_path: str, final_shape, preprocessing: str):
    """
    Load an image, process it and convert it to a friendly tensorflow format
    Args:
        img_path: path of the image to load
        final_shape: Shape that the image will have after the preprocessing. For instance (224, 224)
        preprocessing: preprocessing to apply one among ["NO", "CENTER_CROP_THEN_SCALE", "SQUARE_MARGIN_THEN_SCALE"]

    Returns:

    """
    if preprocessing not in PREPROCESSING_STRATEGIES:
        logging.error(f"{preprocessing} is not in list {PREPROCESSING_STRATEGIES}")
        raise TypeError()

    # Special case if no preprocessing is needed. In this case, the image can directly be loaded in a str format
    # if preprocessing == "NO":
    #     with open(img_path, 'rb') as f:
    #         return f.read()

    # For the other preprocessing, loading the image in opencv is needed
    img = cv2.imread(img_path)
    shape = img.shape
    if preprocessing == "CENTER_CROP_THEN_SCALE":
        # Center-crop the image
        min_shape = min(shape[0], shape[1])
        crop_beginning = [int((shape[0] - min_shape) / 2),
                          int((shape[1] - min_shape) / 2)]
        img = img[crop_beginning[0]:crop_beginning[0] + min_shape,
                  crop_beginning[1]:crop_beginning[1] + min_shape, :]
    elif preprocessing == "SQUARE_MARGIN_THEN_SCALE":
        max_shape = max(shape[0], shape[1])
        new_image = np.zeros((max_shape, max_shape, 3), dtype=np.uint8)
        upper_left_point = [int((max_shape - shape[i]) / 2) for i in range(2)]

        a = upper_left_point[0]
        b = upper_left_point[1]
        new_image[a: a + shape[0], b: b + shape[1], :] = img
        img = new_image
    # Resize the image
    if preprocessing != "NO":
        img = cv2.resize(img, final_shape)
        shape = final_shape
    img_str = cv2.imencode('.jpg', img)[1].tobytes()
    return img_str, shape[0], shape[1]

def store_images_in_tfrecords(images: List[dict], tfrecord_dir_path: str, tfrecord_prefix: str, tfrecord_size: int,
                              preprocessing="NO", image_size=(224, 224)):
    """
        store images in tf records
    Args:
        images: list of image info, each contain path and class id
        tfrecord_dir_path: directory path to write the tfrecords
        tfrecord_prefix: dataset split name
        tfrecord_size: number of images to write in a tfrecord file
        preprocessing:  one of PREPROCESSING_STRATEGIES
        image_size: size of the image to load in the tfrecord
    """

    tfrecord_manager = TfRecordManager(tfrecord_dir_path, tfrecord_prefix, tfrecord_size)
    # Store all images in tfrecord
    for i, image_info in enumerate(images):
        try:
            img, height, width = load_and_preprocess_img(image_info['path'], image_size, preprocessing)
            feature = {'image': bytes_feature(img), 'label': int64_feature(image_info['id']),
                       'size': int64_list_feature([height, width])}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            tfrecord_manager.add(example)
        except Exception as e:
            logging.warning('error with image', image_info['path'], ':', e)

    print("Done")

class UpStrideDatasetBuilderClassification:
    def __init__(self, name: str, description: str, tfrecord_dir_path: str, splits: List[Split], tfrecord_size=10000,
                 preprocessing="NO", image_size=(224, 224)):
        """ Dataset builder class

            Args:
                name: name of the dataset
                description: description of the dataset
                tfrecord_dir_path: directory path to write the tfrecords
                splits:
                tfrecord_size: number of images to write in a tfrecord file
                preprocessing: one of PREPROCESSING_STRATEGIES
                image_size: (224, 224) the size of the image to load in the tfrecord
        """
        self.metadata = {"name": name, "description": description, 'splits': {}}
        self.tfrecord_dir_path = os.path.join(tfrecord_dir_path, name)
        self.splits = splits
        self.tfrecord_size = tfrecord_size
        self.preprocessing = preprocessing
        assert len(image_size) == 2, f"image_size must have a len of 2"
        self.image_size = image_size

    def __tfrecord_files_list(self, split_name):
        """
        list all the tfrecords file in the tfrecord_dir_path for a particular dataset split
        Args
            split_name: split name
        """
        return [os.path.basename(path) for path in
                glob.glob(os.path.join(self.tfrecord_dir_path, split_name + "*.tfrecord"))]

    def __tfrecord_from_annotation_file(self, split: Split):
        """ this function create tfrecord from annotation file

        Args:
            split: Split object which defines the attributes of splits
        """

        print(f"Creating {split.name} split.....")

        if not os.path.exists(split.annotation_file_path):
            raise FileNotFoundError(f"Provided annotation file {split.annotation_file_path} does not exist!!!")

        with open(split.annotation_file_path, 'r') as f:
            data = f.readlines()
            
        if split.header_exists:
            data = data[1:] # ignore first record 

        collection = {}

        for line in data:
            line_item = line.split(',') 
            collection[line_item[0].strip()] =  line_item[1].strip() # key: image file, value: class name

        classes_names = list(set(collection.values()))
        n_classes = len(classes_names)
        print(f"Found {n_classes} classes")


        images = []
        # Now walk through the several dir to find all the images
        for root, dirs, files in os.walk(os.path.join(split.images_dir_path)):
            for f in files:
                # Look for only the image files
                if f.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                    i = classes_names.index(collection[f])
                    images.append({'path': os.path.join(split.images_dir_path, root, f),
                                   'id': i})
        random.shuffle(images)

        store_images_in_tfrecords(images, self.tfrecord_dir_path, split.name, self.tfrecord_size,
                                  self.preprocessing, self.image_size)

        self.metadata['splits'][split.name] = {"tfrecord_files": self.__tfrecord_files_list(split.name),
                                               "num_examples": len(images)}
        self.metadata["classes_names"] = classes_names

    def __tfrecord_from_dirs(self, split: Split):
        """ this function create tfrecord from directory where images are stored in the sub directories of
            `annotation_file_path` by the name of their classes (labels)

        Args:
            split: Split object which defines the attributes of splits
        """

        print(f"Creating {split.name} split.....")

        # Find all the dirs containing the images
        classes_names = os.listdir(split.images_dir_path)
        classes_names = list(filter(lambda x: os.path.isdir(os.path.join(split.images_dir_path, x)), classes_names))
        classes_names.sort()
        n_classes = len(classes_names)
        print(f"Found {n_classes} classes")

        # Now walk through the several dir to find all the images
        images = []
        for i, class_name in enumerate(classes_names):
            for root, dirs, files in os.walk(os.path.join(split.images_dir_path, class_name)):
                for f in files:
                    # Look for only the image files
                    if f.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                        images.append({'path': os.path.join(split.images_dir_path, class_name, root, f),
                                       'id': i})
        random.shuffle(images)

        store_images_in_tfrecords(images, self.tfrecord_dir_path, split.name, self.tfrecord_size,
                                  self.preprocessing, self.image_size)

        self.metadata['splits'][split.name] = {"tfrecord_files": self.__tfrecord_files_list(split.name),
                                               "num_examples": len(images)}
        self.metadata["classes_names"] = classes_names

    def __store_dataset_metadata(self):
        """
        Store the dataset metadata in the tf record directory

        """
        with open(os.path.join(self.tfrecord_dir_path, 'dataset_info.yaml'), 'w') as f:
            yaml.dump(self.metadata, f, default_flow_style=False, sort_keys=False)

    def build(self):
        for split in self.splits:
            if split.annotation_file_path is None:
                self.__tfrecord_from_dirs(split)
            else:
                self.__tfrecord_from_annotation_file(split)

        self.__store_dataset_metadata()


def build_tfrecord_dataset(args):
    # Find all splits to process
    splits = []
    for split_name in ['train', 'validation', 'test']:
        split_data = args[split_name]
        if split_data["images_dir_path"] == '':
            continue
        splits.append(Split(name=split_name, images_dir_path=split_data["images_dir_path"],
                            annotation_file_path=split_data["annotation_file_path"],
                            delimiter=split_data["delimiter"], header_exists=split_data["header_exists"]))
    if not splits:
        raise ValueError("Image Dataset directory is not provided for the splits!")
    if args["tfrecord_dir_path"] == '':
        raise ValueError("Directory for storing tf records is not provided!")
    if args["name"] == '':
        raise ValueError("Dataset name is not provided!")

    builder = UpStrideDatasetBuilderClassification(args["name"], args["description"], args["tfrecord_dir_path"],
                                     splits, args["tfrecord_size"])

    builder.build()


dataset_arguments = [
    [str, 'name', "", 'Name of the dataset'],
    [str, 'description', "", 'Description of the dataset'],
    [str, 'tfrecord_dir_path', "", 'Directory where to store tfrecords'],
    [int, 'tfrecord_size', 1000, 'Number of images to be stored for each file'],
    ['namespace', 'train', [
        [str, 'images_dir_path', '', ' directory path for the images'],
        [str, 'annotation_file_path', None, 'annotation file path in the format `1st column`: images names, `2nd column`: label. '
         'if it is `None` it will be assumed that images are stored in the sub directories of `images_dir_path`'
         'by the name of their classes'],
        [str, 'delimiter', ',', 'Delimiter to split the annotation file columns'],
        [bool, 'header_exists', False, 'whether there is any header in the annotation file'],
    ]],
    ['namespace', 'validation', [
        [str, 'images_dir_path', '', ' directory path for the images'],
        [str, 'annotation_file_path', None, 'annotation file path in the format `1st column`: images names, `2nd column`: label. '
         'if it is `None` it will be assumed that images are stored in the sub directories of `images_dir_path`'
         'by the name of their classes'],
        [str, 'delimiter', ',', 'Delimiter to split the annotation file columns'],
        [bool, 'header_exists', False, 'whether there is any header in the annotation file'],
    ]],
    ['namespace', 'test', [
        [str, 'images_dir_path', '', ' directory path for the images'],
        [str, 'annotation_file_path', None, 'annotation file path in the format `1st column`: images names, `2nd column`: label. '
         'if it is `None` it will be assumed that images are stored in the sub directories of `images_dir_path`'
         'by the name of their classes'],
        [str, 'delimiter', ',', 'Delimiter to split the annotation file columns'],
        [bool, 'header_exists', False, 'whether there is any header in the annotation file'],
    ]]
]

if __name__ == "__main__":
    args = parse_cmd(dataset_arguments)
    build_tfrecord_dataset(args)
