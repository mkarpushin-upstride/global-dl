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
training_arguments_base = [
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
        [bool, "xla", False, "if true then use xla"],
        [bool, "mirrored", False, 'If true then use mirrored strategy'],
        [bool, "with_mixed_precision", False, 'To train with mixed precision'],
        [bool, 'profiler', False, 'if true then profile tensorflow training using tensorboard. Need tf >=2.2'],
    ]],
    ['list[int]', "input_size", [224, 224, 3], 'processed shape of each image'],
    ['namespace', 'lr_decay_strategy', [
        [bool, 'activate', True, 'if true then use this callback'],
        ['namespace', 'lr_params', [
            [str, 'strategy', 'lr_reduce_on_plateau', 'learning rate decay schedule', lambda x: x.lower() in learning_schedule_list],
            [int, 'power', 5, 'used only in polynomial_decay, determines the nth degree polynomial'],
            [float, 'alpha', 0.01, 'used only in cosine decay, Minimum learning rate value as a fraction of initial_learning_rate. '],
            [int, 'patience', 10, 'used only in lr_reduce_on_plateau, if validation loss doesn\'t improve for this number of epoch, then reduce the learning rate'],
            [float, 'decay_rate', 0.5, 'used step_decay, step_decay_schedule, inverse_time_decay, lr_reduce_on_plateau, determines the factor to drop the lr'],
            [float, 'min_lr', 0.00001, 'usef in lr_reduce_on_plateau'],
            [int, 'early_stopping', 20, 'stop  the training if validation loss doesn\'t decrease for n value'],
            [int, 'drop_after_num_epoch', 10, 'used in step_decay, reduce lr after specific number of epochs'],
            ['list[int]', 'drop_schedule', [30, 50, 70], 'used in step_decay_schedule, reduce lr after specific number of epochs'],
            [float, 'decay_step', 1.0, 'used in inverse time decay, decay_step controls how fast the decay rate reduces '],
            [bool, 'staircase', False, 'if true then return the floor value of inverse_time_decay'],
        ]],
    ]],
    ['namespace', 'data_aug_param', [
        ['list[str]', 'train_list', ['RandomCropThenResize', 'RandomHorizontalFlip', 'Normalize'], 'List all the data augmentations separated by comma for training data'],
        # TODO change validation operations to crop the biggest possible square in the image and then resize
        ['list[str]', 'val_list', ['Resize', 'CentralCrop', 'Normalize'], 'List all the data augmentations separated by comma for validation data'],
        ['namespace', 'Normalize', [
            ['list[float]', "mean", [0.485, 0.456, 0.406], 'mean of training data'],
            ['list[float]', "std", [0.229, 0.224, 0.225], 'std of training data'],
            [bool, "scale_in_zero_to_one", True, 'only scale the image in the  range (0~1)'],
            [bool, "only_subtract_mean", False, 'if True, subtract only mean from input without dividing by std']
        ]],
        ['namespace', 'ColorJitter', [
            [float, "brightness", 0.05, 'brightness factor to jitter'],
            ['list[float]', "contrast", [0.7, 1.3], 'contrast range to jitter'],
            ['list[float]', "saturation", [0.6, 1.6], 'saturation range to jitter'],
            [float, "hue", 0.08, 'hue factor to jitter'],
            ['list[float]', "clip", [0., 1.0], 'clipping range, if both (min, max) are 0, no clipping will be performed']
        ]],
        ['namespace', 'RandomRotate', [
            [int, "angle", 10, 'angle will be selected randomly within range [-angle, angle]'],
            [str, "interpolation", 'nearest', 'interpolation method']
        ]],
        ['namespace', 'CentralCrop', [
            ['list[int]', "size", [224, 224], 'size of the central crop'],
            [float, "crop_proportion", 0.875, 'proportion of image to retain along the less-cropped side'],
            [str, "interpolation", 'bicubic', 'interpolation method']
        ]],
        ['namespace', 'RandomCrop', [
            ['list[int]', "size", [224, 224, 3], 'Random crop shape']
        ]],
        ['namespace', 'Resize', [
            ['list[int]', "size", [224, 224], 'shape for resizing the image'],
            [str, "interpolation", 'bicubic', 'interpolation method']
        ]],
        ['namespace', 'ResizeThenRandomCrop', [
            ['list[int]', "size", [256, 256], 'shape for resizing the image'],
            ['list[int]', "crop_size", [224, 224, 3], 'Random crop shape'],
            [str, "interpolation", 'bicubic', 'interpolation method']
        ]],
        ['namespace', 'RandomCropThenResize', [
            ['list[int]', "size", [224, 224], 'shape for resizing the image'],
            ['list[float]', "scale", [0.08, 1.0], 'range of size of the origin size cropped'],
            ['list[float]', "ratio", [0.75, 1.33], 'range of aspect ratio of the origin aspect ratio cropped'],
            [str, "interpolation", 'bicubic', 'interpolation method']
        ]],
        ['namespace', 'Translate', [
            [float, "width_shift_range", 0.1, 'randomly shift images horizontally (fraction of total width)'],
            [float, "height_shift_range", 0.1, 'randomly shift images vertically (fraction of total height)']
        ]]
    ]],
    ['namespace', 'tensorrt', [
        [bool, 'export_to_tensorrt', False, 'If specified, converts the savemodel to TensorRT and saves within export_dir'],
        [str, 'precision_tensorrt', 'FP16', 'Optimizes the TensorRT model to the precision specified'],
    ]],
    ['namespace', 'server', [
        [str, 'user', '', 'Username to connect to upstride platform'],
        [str, 'password', '', 'Password to connect to upstride platform'],
        [str, 'id', '', 'id of the training, provided by the server at the first post request'],
        [str, 'jwt', '', 'javascript web token for auth']
    ]],
    [str, 'tf_dataset_name', 'imagenet', 'Choose the dataset to be used for training'],
    [str, 'custom_data_dir', '', 'If not using tf_data, directory dataset is stored'],
    ['namespace', 'imagenet_data', [
        [str, "synset_path", 'LOC_synset_mapping.txt', 'Imagnet synset mapping file'],
        [str, "train_dir", '', 'Directory where are training data'],
        [str, "val_dir", '', 'Directory where are val data'],
        [str, "val_gt_path", 'LOC_val_solution.csv', 'Validation data ground-truth file'],
        [int, "train_data_percentage", 100, 'Percentage of data to be used for training the UpStride model'],
    ]]
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
  callbacks = [tensorboard_cb, tf.keras.callbacks.EarlyStopping('val_loss', patience=args['lr_decay_strategy']['lr_params']['early_stopping'])]
  if args['lr_decay_strategy']['activate']:
    callbacks.append(
        get_lr_scheduler(args['lr'], args['num_epochs'], args['lr_decay_strategy']['lr_params'])
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
