import os
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from .optimizers import get_lr_scheduler

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
  if args['with_mixed_precision']:
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
  if args['mirrored']:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
      model, optimizer = get_model(args)
  else:
    model, optimizer = get_model(args)
  return model, optimizer

def get_callbacks(args, log_dir):
  # define callbacks
  if args['profiler']:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False, profile_batch='10, 12')
  else:
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False, profile_batch=0)
  callbacks = [tensorboard_cb, tf.keras.callbacks.EarlyStopping('val_loss', patience=args['lr_decay_strategy']['lr_params']['early_stopping'])]
  if args['lr_decay_strategy']['activate']:
    callbacks.append(
        get_lr_scheduler(args['lr'], args['num_epochs'], args['lr_decay_strategy']['lr_params'])
    )
  return callbacks
