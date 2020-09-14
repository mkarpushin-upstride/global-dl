import os
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt


def convert_to_tensorrt(model_path, image_size, batch_size, precision='FP16'):
  precision_list = ['FP16', 'FP32', 'INT8']
  precision_value = precision.upper()
  if precision_value in precision_list:
    folder = f'TensorRT_{precision_value}'
  else:
    raise ValueError(f'value {precision} not found, must be one of these {precision_list}')

  conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
  conversion_params = conversion_params._replace(
      max_workspace_size_bytes=12000000000,  # 8 gigs
      precision_mode=precision_value,
      maximum_cached_engines=1,
  )
  converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_path,
                                      conversion_params=conversion_params)

  def my_input_fn():
    image_dimensions = [image_size[:-1]]  # get [224, 224]
    for dims in image_dimensions:
      inp = np.random.normal(size=(batch_size, *dims, 3)).astype(np.float32)
      yield [inp]

  if precision_value == 'INT8':
    # TODO need to pass matrix with similar data distribution as trained.
    # Ideally samples from validation set for all classes so that rescaling
    # and threshold optimization for INT8 is computed correctly without significant loss
    raise NotImplementedError('INT8 functionality is not available yet.')
  else:
    converter.convert()

  tensorrt_output_path = os.path.join(model_path, folder)
  converter.build(input_fn=my_input_fn)
  converter.save(tensorrt_output_path)
  print(f"converted model present at {tensorrt_output_path}")

  return tensorrt_output_path
