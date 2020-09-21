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
  if config['tensorrt']['export_to_tensorrt']:
    convert_to_tensorrt(
        path,
        image_size=config['processed_size'],
        batch_size=config['batch_size'],
        precision=config['tensorrt']['precision_tensorrt'])

  # creates the tar file
  print('compress')
  model_tar = os.path.join(tmp_dir, "model.tar.gz")
  with tarfile.open(model_tar, "w:gz") as tar:
    tar.add(path)
  print('compress done')

  if 'upstride' in config['export_strategy_cloud']:
    upstride(model_tar, config)

  shutil.rmtree(tmp_dir)  # remove local model directory
  logger.info("Model saved Successfully")
