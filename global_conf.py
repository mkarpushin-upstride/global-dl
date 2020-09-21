import tensorflow as tf


def config_tf2(use_xla=False):
  """ By default tensorflow 2 take the whole memory of the GPU. For shared server, we should change this configuration

  Args:
      use_xla (bool, optional): In some cases, using xla can speed up training or inference. Defaults to False.
  """
  if use_xla:
    tf.config.optimizer.set_jit(True)

  physical_devices = tf.config.list_physical_devices('GPU')
  for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)
