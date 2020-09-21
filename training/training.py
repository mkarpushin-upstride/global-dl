import os
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from .optimizers import get_lr_scheduler
from submodules.global_dl.training.export import export_strategies
from submodules.global_dl.training.optimizers import learning_schedule_list


def create_dir(path: str):
  """this function exists to be called by the argument parser,
  to automaticaly create new directories
  """
  try:
    os.makedirs(path, exist_ok=True)
  except FileExistsError as e:
    # this error shouldn't happen because of exist_ok=True, but we never know
    return False
  except FileNotFoundError as e:
    return False
  return True


def create_dir_or_empty(path: str):
  """this function exists to be called by the argument parser,
  to automaticaly create new directories if path is not empty
  """
  if path == "":
    return True
  return create_dir(path)


# list of [type, name, default, help, condition] or ['namespace', name, List]
# if condition is specified and false, then raise an exception
# type can be :
#  - one of the following python types : int, str, bool, float
#  - 'list[{type}]' with type in [int, str, bool, float] (for instance 'list(str)')
#  - 'namespace' to define namespace
arguments = [
    ['list[str]', "yaml_config", [], "config file overriden by these argparser parameters"],
    [str, "checkpoint_dir", '', 'Checkpoints directory', create_dir],
    [str, 'title', '', 'title of the experiment'],
    [str, 'description', '', 'description of the experiment'],
    [str, 'export_dir', '', 'If specified, export the model in this directory', create_dir_or_empty],
    [str, 'export_strategy_cloud', 'upstride', 'to define how to export on the cloud', lambda x: x in export_strategies],
    [str, "log_dir", '', 'Log directory', create_dir],
    [int, "num_classes", 0, 'Number of classes', lambda x: x > 0],  # TODO this number should be computed from dataset
    [int, "num_epochs", 60, 'The number of epochs to run', lambda x: x > 0],
    ['namespace', 'configuration', [
        [bool, "mirrored", False, 'If true then use mirrored strategy'],
        [bool, "with_mixed_precision", False, 'To train with mixed precision'],
        [bool, 'profiler', False, 'if true then profile tensorflow training using tensorboard. Need tf >=2.2'],
    ]],
    ['list[int]', "input_size", [224, 224, 3], 'processed shape of each image'],
    
    ['namespace', 'tensorrt', [
        [bool, 'export_to_tensorrt', False, 'If specified, converts the savemodel to TensorRT and saves within export_dir'],
        [str, 'precision_tensorrt', 'FP16', 'Optimizes the TensorRT model to the precision specified'],
    ]],
    [int, 'early_stopping', 20, 'stop  the training if validation loss doesn\'t decrease for n value'],
]


def create_env_directories(args, experiment_name):
  """
  Args:
      args ([type]): dict need to have checkpoint_dir, log_dir, export_dir
      experiment_name ([type]): [description]

  Returns:
      [type]: [description]
  """
  checkpoint_dir = os.path.join(args['checkpoint_dir'], experiment_name)
  log_dir = os.path.join(args['log_dir'], experiment_name)
  export_dir = os.path.join(args['export_dir'], experiment_name) if args['export_dir'] else None
  return checkpoint_dir, log_dir, export_dir


def setup_mp(args):
  if args['configuration']['with_mixed_precision']:
    print('Training with Mixed Precision')
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print(f'Compute dtype: {policy.compute_dtype}')
    print(f'Variable dtype: {policy.variable_dtype}')
    print(f'Loss scale: {policy.loss_scale}')
    # the LossScaleOptimizer is not needed because model.fit already handle this. See https://www.tensorflow.org/guide/keras/mixed_precision
    # for more information. I let the code here to remember if one day we go to custom training loop
    # opt = mixed_precision.LossScaleOptimizer(opt, loss_scale=policy.loss_scale)


def define_model_in_strategy(args, get_model):
  if args['configuration']['mirrored']:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
      model, optimizer = get_model(args)
  else:
    model, optimizer = get_model(args)
  return model, optimizer


def get_callbacks(args, log_dir):
  # define callbacks
  if args['configuration']['profiler']:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False, profile_batch='10, 12')
  else:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False, profile_batch=0)
  callbacks = [tensorboard_cb, tf.keras.callbacks.EarlyStopping('val_loss', patience=args['early_stopping'])]
  if args['optimizer']['lr_decay_strategy']['activate']:
    callbacks.append(
        get_lr_scheduler(args['optimizer']['lr'], args['num_epochs'], args['optimizer']['lr_decay_strategy']['lr_params'])
    )
  return callbacks


def init_custom_checkpoint_callbacks(trackable_objects, ckpt_dir):
  checkpoint = tf.train.Checkpoint(**trackable_objects)
  manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir, max_to_keep=5)
  latest = manager.restore_or_initialize()
  latest_epoch = 0
  if latest is not None:
    print(f'restore {manager.latest_checkpoint}')
    latest_epoch = int(manager.latest_checkpoint.split('-')[-1])
  return tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: manager.save()), latest_epoch
