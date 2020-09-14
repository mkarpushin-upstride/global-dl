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

def store_object_detection_images_in_tfrecords(collection: List[dict], class_map: dict, tfrecord_dir_path: str, tfrecord_prefix: str, tfrecord_size: int,
                              preprocessing="NO", image_size=(416, 416)):
    """
        store images in tf records
    Args:
        collection: list of image info, each contain path, bounding boxes and  class id
        class_map: dictionary with key as class name and value as unique id. 
        tfrecord_dir_path: directory path to write the tfrecords
        tfrecord_prefix: dataset split name
        tfrecord_size: number of images to write in a tfrecord file
        preprocessing:  TODO one of PREPROCESSING_STRATEGIES.
        image_size: size of the image to load in the tfrecord
    """

    tfrecord_manager = TfRecordManager(tfrecord_dir_path, tfrecord_prefix, tfrecord_size)
    # Store all images in tfrecord
    for i, image_info in enumerate(collection):
        try:
            # TODO preprocessing specific to object detection. 
            img = cv2.imread(image_info['image_path'])
            height, width = img.shape[:-1] # get h, w
            encoded_image_data = cv2.imencode('.jpg', img)[1].tobytes()
            filename=image_info['image_name'].encode("utf8")
            xmins=[]
            xmaxs=[]
            ymins=[]
            ymaxs=[]
            classes_text=[]
            classes=[]

            for label in image_info["labels"]:
                xmins.append(label["xmin"] / width)
                xmaxs.append(label["xmax"] / width)
                ymins.append(label["ymin"] / height)
                ymaxs.append(label["ymax"] / height)
                classes_text.append(label["class"].encode("utf8"))
                classes.append(class_map[label["class"]])

            example = tf.train.Example(features=tf.train.Features(feature={
                    'image/height': int64_feature(height),
                    'image/width': int64_feature(width),
                    'image/filename': bytes_feature(filename),
                    'image/encoded': bytes_feature(encoded_image_data),
                    # 'image/format': bytes_feature(image_format),
                    'image/object/bbox/xmin': float_list_feature(xmins),
                    'image/object/bbox/xmax': float_list_feature(xmaxs),
                    'image/object/bbox/ymin': float_list_feature(ymins),
                    'image/object/bbox/ymax': float_list_feature(ymaxs),
                    'image/object/class/text': bytes_list_feature(classes_text),
                    'image/object/class/label': int64_list_feature(classes),
            }))

            tfrecord_manager.add(example)
        except Exception as e:
            logging.warning('error with image', image_info['image_path'], ':', e)

    print("Done")

class UpStrideDatasetBuilderObjectDetection:
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


    def __object_detection_tfrecord_from_annotation_file(self, split: Split):
        
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

        total_classes = []
        collection = []
        

        for line in data:
            line_item = line.split(' ') 
            get_parsed_data = {
                'image_name':line_item[0].strip(), # get images
                'labels': []
                }
            for items in line_item[1:]: # get bounding box and respective class name
                if len(items) > 1:   # ensure if image doesn't have bounding box and class data its skipped
                    i = items.strip().split(",")
                    get_parsed_data['labels'].append(
                            {
                        "xmin": int(i[0]),
                        "ymin": int(i[1]),
                        "xmax": int(i[2]),
                        "ymax": int(i[3]),
                        "class": i[4]
                        }
                    )
                    total_classes.append(i[4])
            collection.append(get_parsed_data)

        n_classes = len(set(total_classes))
        print(f"Found {n_classes} classes")

        class_map = {name : index for index, name in enumerate(set(total_classes))}

        # Now walk through the several dir to find all the images
        for root, dirs, files in os.walk(os.path.join(split.images_dir_path)):
            for f in files:
                # Look for only the image files
                if f.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                    for get_dict in collection:
                        if get_dict['image_name'] in f:
                            get_dict['image_path'] = os.path.join(split.images_dir_path, root, f)

        random.shuffle(collection)

        store_object_detection_images_in_tfrecords(collection, class_map, self.tfrecord_dir_path, split.name, self.tfrecord_size,
                                    self.preprocessing, self.image_size)

        self.metadata['splits'][split.name] = {"tfrecord_files": self.__tfrecord_files_list(split.name),
                                                "num_examples": len(collection)}
        self.metadata["classes_names"] = list(class_map.values())


    def __store_dataset_metadata(self):
        """
        Store the dataset metadata in the tf record directory

        """
        with open(os.path.join(self.tfrecord_dir_path, 'dataset_info.yaml'), 'w') as f:
            yaml.dump(self.metadata, f, default_flow_style=False, sort_keys=False)

    def build(self):
        for split in self.splits:
            self.__object_detection_tfrecord_from_annotation_file(split)
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

    builder = UpStrideDatasetBuilderObjectDetection(args["name"], args["description"], args["tfrecord_dir_path"],
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
