import tensorflow as tf


arguments = [
    [bool, "xla", False, "In some cases, using xla can speed up training or inference"],
    [bool, "full_gpu_memory", False, "By default, the model will take only what it needs as GPU memory. By turning on this option, it will use the whole GPU memory"]
]


def config_tf2(config):
  """ By default tensorflow 2 take the whole memory of the GPU. For shared server, we should change this configuration

  Args:
      config (dict): dictionary containing 'xla' and 'full_gpu_memory'
  """
  if config['xla']:
    tf.config.optimizer.set_jit(True)
  if not config['full_gpu_memory']:
    physical_devices = tf.config.list_physical_devices('GPU')
    for physical_device in physical_devices:
      tf.config.experimental.set_memory_growth(physical_device, True)
