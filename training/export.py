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

export_strategies = ['aws',  # push model to aws bucket
                     'upstride',  # push model to upstride API
                     '']  # do nothing


def aws(model_tar):
  import boto3
  from botocore.exceptions import NoCredentialsError
  """
    Prior launching the docker ensure credentials are configured via "aws configure"
    and your "/home/$USER/.aws/ is mapped to "/.aws/"
    """
  logger = logging.getLogger("SaveModel")
  os.environ['AWS_SHARED_CREDENTIALS_FILE'] = "/.aws/credentials"
  s3 = boto3.resource("s3")
  bucket = s3.Bucket("modelzoo")

  try:
    logger.info("Trying to upload modelzoo.")
    bucket.upload_file(model_tar, path + ".tar")  # upload tar file to s3
  except NoCredentialsError as err:
    logger.error(err)
    logger.info("S3 upload failed, Model available locally at {}".format(path))
  # TODO handle other types of exceptions later.


def upstride(archive_path, args):
  """call the main swagger upstride api to define a new experiment result and push the model
  """
  send_exported_model(args, archive_path)


def export(model, path, args):
  logger = logging.getLogger("SaveModel")
  tmp_dir = tempfile.mkdtemp()
  if path is None:
    path = tmp_dir

  tf.saved_model.save(model, path)
  # WIP - use export_to_tensorrt: False for now
  if args['tensorrt']['export_to_tensorrt']:
    convert_to_tensorrt(
        path,
        image_size=args['processed_size'],
        batch_size=args['batch_size'],
        precision=args['tensorrt']['precision_tensorrt'])

  # creates the tar file
  print('compress')
  model_tar = os.path.join(tmp_dir, "model.tar.gz")
  with tarfile.open(model_tar, "w:gz") as tar:
    tar.add(path)
  print('compress done')

  if 'aws' in args['export_strategy_cloud']:
    aws(model_tar)
  if 'upstride' in args['export_strategy_cloud']:
    upstride(model_tar, args)

  shutil.rmtree(tmp_dir)  # remove local model directory
  logger.info("Model saved Successfully")
