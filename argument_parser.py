import argparse
import yaml
import os

# from .models import model_name_to_class
# from .models.generic_model import framework_list
# from .export import export_strategies
# from .model_tools import optimizer_list, learning_schedule_list


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
training_arguments = [
    [int, "batch_size", 128, 'The size of batch per gpu', lambda x: x > 0],
    [str, "checkpoint_dir", '', 'Checkpoints directory', create_dir],
    [bool, "cpu", False, 'create model on cpu'],
    [str, 'dataset_name', 'imagenet', 'Choose the dataset to be used for training'],
    [str, 'data_dir', '', 'Directory datset is stored'],
    [str, 'description', '', 'description of the experiment'],
    [int, 'early_stopping', 6, 'Early stopping patience'],
    [str, 'export_dir', '', 'If specified, export the model in this directory. Usefull only if kerastuner is False', create_dir_or_empty],
    # [str, 'export_strategy_cloud', '', 'to define how to export on the cloud', lambda x: x in export_strategies],
    [int, "factor", 1, 'factor by which UpStride model number of channels should decrease'],
    # [str, 'framework', 'tensorflow', 'Framework to use to define the model', lambda x: x in framework_list],
    [bool, 'kerastuner', False, 'use keras tuner instead of keras'],
    [str, "log_dir", '', 'Log directory', create_dir],
    [float, "lr", 0.0001, 'learning rate', lambda x: x > 0],
    # [str, "model_name", '', 'Specify the name of the model', lambda x: x in model_name_to_class],
    [int, 'n_layers_before_tf', 0, 'when using mix framework, number of layer defined using upstride', lambda x: x >= 0],
    [int, "num_classes", 0, 'Number of classes', lambda x: x > 0],  # TODO this number should be computed from dataset
    [int, "num_epochs", 60, 'The number of epochs to run', lambda x: x > 0],
    [bool, "mirrored", False, 'If true then use mirrored strategy'],
    ['list[int]', "processed_size", [224, 224, 3], 'processed shape of each image'],
    [bool, 'profiler', False, 'if true then profile tensorflow training using tensorboard. Need tf >=2.2'],
    ['list[int]', "raw_size", [256, 256, 3], 'raw shape of each image'],
    [str, 'title', '', 'title of the experiment'],
    ['list[str]', "yaml_config", [], "config file overriden by these argparser parameters"],
    [bool, "with_mixed_precision", False, 'To train with mixed precision'],
    [bool, "xla", False, "if true then use xla"],
    # ['namespace', 'lr_decay_strategy', [
    #     [bool, 'activate', False, 'if true then use this callback'],
    #     ['namespace', 'lr_params', [
    #         [str, 'strategy', '', 'learning rate decay schedule', lambda x: x.lower() in learning_schedule_list],
    #         [int, 'patience', 4, 'if validation loss doesn\'t improve for this number of epoch, then reduce the learning rate'],
    #         [float, 'factor', 0.2, ''],
    #         [float, 'min_lr', 0.00001, ''],
    #         [int, 'early_stopping', 10, 'stop  the training if validation loss doesn\'t decrease for n value'],
    #         [float, 'drop_rate', 0.5, 'used in step_decay, determines the factor to drop the lr'],
    #         [int, 'drop_after_num_epoch', 10, 'used in step_decay, reduce lr after specific number of epochs'],
    #         [int, 'power', 5, 'used in polynomial_decay, determines the nth degree polynomial'],
    #         [float, 'decay_rate', 0.5, 'used in inverse time decay, decay_rate is multiplied to epoch'],
    #         [float, 'decay_step', 1.0, 'used in inverse time decay, decay_step controls how fast the decay rate reduces '],
    #         [float, 'alpha', 0.01, 'used in cosine decay, Minimum learning rate value as a fraction of initial_learning_rate. '],
    #         [bool, 'staircase', False, 'if true then return the floor value of inverse_time_decay'],
    #     ]],
    # ]],
    # ['namespace', 'optimizer_param', [
    #     [str, 'name', 'sgd_nesterov', 'optimized to be used', lambda x: x.lower() in optimizer_list],
    #     [float, 'momentum', 0.9, 'used when optimizer name is specified as sgd_momentum'],
    # ]],
    ['namespace', 'data_aug_param', [
        ['list[str]', 'train_list', ['ResizeThenRandomCrop', 'RandomHorizontalFlip', 'Normalize'], 'List all the data augmentations separated by comma for training data'],
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
    ['namespace', 'server', [
        [str, 'user', '', 'Username to connect to upstride platform'],
        [str, 'password', '', 'Password to connect to upstride platform'],
        [str, 'id', '', 'id of the training, provided by the server at the first post request'],
        [str, 'jwt', '', 'javascript web token for auth']
    ]],
    ['namespace', 'imagenet_data', [
        [str, "synset_path", 'LOC_synset_mapping.txt', 'Imagnet synset mapping file'],
        [str, "train_dir", '', 'Directory where are training data'],
        [str, "val_dir", '', 'Directory where are val data'],
        [str, "val_gt_path", 'LOC_val_solution.csv', 'Validation data ground-truth file'],
        [int, "train_data_percentage", 100, 'Percentage of data to be used for training the UpStride model'],
    ]],
    ['namespace', 'tensorrt', [
        [bool, 'export_to_tensorrt', False, 'If specified, converts the savemodel to TensorRT and saves within export_dir'],
        [str, 'precision_tensorrt', 'FP16', 'Optimizes the TensorRT model to the precision specified'],
    ]]
]


def str2bool(v):
  """idea from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
  """
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def create_argparse(arguments, namespace='', parser=None):
  # init parser at beginning
  init = False
  if parser is None:
    init = True
    parser = argparse.ArgumentParser(description="Upstride implementation of Models")

  for argument in arguments:
    # see https://docs.python.org/3/library/argparse.html#nargs for use of nargs='?'
    # check if namespace
    if argument[0] == 'namespace':
      full_namespace = f'{namespace}.{argument[1]}' if namespace != '' else argument[1]
      create_argparse(argument[2], namespace=full_namespace, parser=parser)
      continue
    arg_name = f'--{namespace}.{argument[1]}' if namespace != '' else f'--{argument[1]}'
    help = f'{argument[3]} [default: {argument[2]}]'
    if argument[0] == bool:
      parser.add_argument(arg_name, type=str2bool, nargs='?', const=True, help=help)
      continue
    if argument[0] in [int, str, float]:
      parser.add_argument(arg_name, type=argument[0],  nargs='?', help=help)
      continue

    # at this point, argument[0] should be a string containing 'list'
    if type(argument[0]) != str or 'list' not in argument[0]:
      raise TypeError(f'{argument[0]} is not a correct data type')

    custom_type = argument[0].split('[')[1][:-1]
    if custom_type == 'bool':
      parser.add_argument(arg_name, type=str2bool, nargs='*', help=help)
      continue
    if custom_type in ['int', 'str', 'float']:
      d = {
          'int': int,
          'str': str,
          'float': float
      }
      custom_type = d[custom_type]
      parser.add_argument(arg_name, type=custom_type, nargs='*', help=help)
      continue

    raise TypeError(f'{argument[0]} is not a correct data type')
  if init:
    return parser.parse_args()


def init_parameters(arguments):
  # define the parameter dict with all values to None
  parameters = {}
  for argument in arguments:
    if argument[0] != 'namespace':
      parameters[argument[1]] = None
    else:
      sub_parameters = init_parameters(argument[2])
      parameters[argument[1]] = sub_parameters
  return parameters


def merge_dict(parameters, arguments):
  for key in arguments:
    if parameters[key] is None:
      parameters[key] = arguments[key]
    elif type(parameters[key]) == dict:
      merge_dict(parameters[key], arguments[key])
    else:
      raise Exception("this line shouldn't be excecuted, please check the code")


def read_yaml_config(yaml_file, parameters):
  with open(yaml_file, 'r') as file:
    content = yaml.safe_load(file)
    merge_dict(parameters, content)


def check_and_add_defaults(arguments, parameters):
  # Lastly, if a variable is defined nowhere, then use the default value
  for argument in arguments:
    if argument[0] == 'namespace':
      if argument[1] not in parameters:
        parameters[argument[1]] = {}
      parameters[argument[1]] = check_and_add_defaults(argument[2], parameters[argument[1]])
    else:
      if argument[1] not in parameters or parameters[argument[1]] == None:
        parameters[argument[1]] = argument[2]
      # Now check for additional conditions
      if len(argument) == 5:
        if not argument[4](parameters[argument[1]]):
          raise ValueError("condition for parameter {} not satisfied by value {}".format(argument[1], parameters[argument[1]]))
  return parameters


def parse_cmd(arguments):
  # init parameters dict, read command line and conf file
  parameters = init_parameters(arguments)
  args = create_argparse(arguments)
  if "yaml_config" in vars(args) and vars(args)["yaml_config"] is not None:
    for conf_file in args.yaml_config:
      read_yaml_config(conf_file, parameters)
  # Overwrite config using args
  for key in vars(args):
    if key == "yaml_config" or vars(args)[key] is None:
      continue
    parameters = add_value_in_param(parameters, key, vars(args)[key])
  parameters = check_and_add_defaults(arguments, parameters)
  return parameters


def add_value_in_param(parameters, key, value):
  """
  """
  if '.' not in key:
    parameters[key] = value
  else:
    key_split = key.split('.')
    parameters[key_split[0]] = add_value_in_param(parameters[key_split[0]], '.'.join(key_split[1:]), value)
  return parameters
