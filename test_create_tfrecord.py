import os
import shutil
import tempfile
import unittest
from cv2 import cv2
import numpy as np
from create_tfrecord_classification import UpStrideDatasetBuilderClassification
from create_tfrecord_object_detection import UpStrideDatasetBuilderObjectDetection


from create_tf_utils import Split


class TestCreateTfrecord(unittest.TestCase):
    def test_process(self):
        train_dir = create_fake_dataset()
        test_functions = {
            UpStrideDatasetBuilderClassification: create_fake_dataset_with_annotation_file,
            UpStrideDatasetBuilderObjectDetection: create_fake_dataset_with_od_annotation_file,
        }
        for dataset_build, function in test_functions.items():
            val_dir, val_annotation_file = function()
            name = 'Test-dataset'
            description = 'A small test datset'
            tfrecord_dir_path = tempfile.mkdtemp()
            splits = [Split(name='train', images_dir_path=train_dir,
                                annotation_file_path=val_annotation_file, delimiter=',', header_exists=False),
                        Split(name='val', images_dir_path=val_dir,
                                annotation_file_path=val_annotation_file, delimiter=',', header_exists=False)
                        ]
            tfrecord_size = 2
            print(splits[0].annotation_file_path)
            bulider = dataset_build(name, description, tfrecord_dir_path, splits, tfrecord_size)
            bulider.build()

            print(tfrecord_dir_path)
        shutil.rmtree(train_dir)
        shutil.rmtree(val_dir)


def create_fake_dataset(n_images_per_class=2):
    dataset_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(dataset_dir, 'cat'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'dog'), exist_ok=True)
    for i in range(n_images_per_class):
        cv2.imwrite(os.path.join(dataset_dir, 'dog', '{}.jpg'.format(i)), np.ones((640, 480, 3), dtype=np.uint8) * 255)
        cv2.imwrite(os.path.join(dataset_dir, 'cat', '{}.jpg'.format(i)), np.ones((640, 480, 3), dtype=np.uint8) * 255)
    return dataset_dir


def create_fake_dataset_with_annotation_file(n_images_per_class=2):
    dataset_dir = tempfile.mkdtemp()
    os.makedirs(dataset_dir, exist_ok=True)

    annotation_file = os.path.join(dataset_dir, 'annotations.txt')

    labels = ['cat', 'dog']

    with open(annotation_file, 'w', encoding='utf-8') as f:
        for i in range(n_images_per_class*2):
            cv2.imwrite(os.path.join(dataset_dir, '{}.jpg'.format(i)), np.ones((640, 480, 3), dtype=np.uint8) * 255)
            line = '{}.jpg'.format(i) + "," + labels[i%2] + "\n"
            f.write(line)

    return dataset_dir, annotation_file

def create_fake_dataset_with_od_annotation_file(n_images_per_class=2):
    dataset_dir = tempfile.mkdtemp()
    os.makedirs(dataset_dir, exist_ok=True)

    annotation_file = os.path.join(dataset_dir, 'annotations.txt')

    labels = ['cat', 'dog']
    xmin = [40, 35]
    ymin = [200, 202]
    xmax = [100, 96]
    ymax = [300, 250]

    delimiter = " "
    line_return = "\n"

    with open(annotation_file, 'w', encoding='utf-8') as f:
        for i in range(n_images_per_class):
            cv2.imwrite(os.path.join(dataset_dir, '{}.jpg'.format(i)), np.ones((640, 480, 3), dtype=np.uint8) * 255)
            image = '{}.jpg'.format(i) 
            bbox = str(xmin[i%2]) + "," + str(ymin[i%2]) + "," + str(xmax[i%2]) + "," + str(ymax[i%2]) + "," + labels[i%2]
            if i%2 == 0:
                f.write(image + delimiter + bbox + delimiter + line_return)
            else:
                f.write(image + delimiter + bbox + delimiter + bbox + line_return)
    return dataset_dir, annotation_file