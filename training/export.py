"""this file contain everything to read and write tf.saved_model and tf.checkpoint
soon it will also push the model to the zoo when calling export
"""

import logging
import os
import shutil
import tarfile
import tempfile

import tensorflow as tf
from .trt_convert import convert_to_tensorrt
from .alchemy_api import send_exported_model
from .training import create_dir_or_empty

arguments = [
    [str, 'dir', '', 'If specified, export the model in this directory', create_dir_or_empty],
    [str, 'strategy_cloud', '', 'to define how to export on the cloud', lambda x: x in export_strategies],
    ['namespace', 'tensorrt', [
        [bool, 'export', False, 'If specified, converts the savemodel to TensorRT and saves within export_dir'],
        [str, 'precision', 'FP16', 'Optimizes the TensorRT model to the precision specified'],
    ]],
]

export_strategies = ['upstride',  # push model to upstride plateform
                     '']  # do nothing


def upstride(archive_path, config):
  """call the main swagger upstride api to define a new experiment result and push the model
  """
  send_exported_model(config, archive_path)


def export(model, path, config):
  logger = logging.getLogger("SaveModel")
  tmp_dir = tempfile.mkdtemp()
  if path is None:
    path = tmp_dir

  tf.saved_model.save(model, path)
  # WIP - use export_to_tensorrt: False for now
  if config['export']['tensorrt']['export']:
    convert_to_tensorrt(
        path,
        image_size=config['input_size'],
        batch_size=config['dataloader']['batch_size'],
        precision=config['export']['tensorrt']['precision'])

  # creates the tar file
  print('compress')
  model_tar = os.path.join(tmp_dir, "model.tar.gz")
  with tarfile.open(model_tar, "w:gz") as tar:
    tar.add(path)
  print('compress done')

  if 'upstride' in config['export']['strategy_cloud']:
    upstride(model_tar, config)

  shutil.rmtree(tmp_dir)  # remove local model directory
  logger.info("Model saved Successfully")
