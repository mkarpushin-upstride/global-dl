import shutil
import tempfile
import unittest
import tensorflow as tf
import numpy as np
from .export import export
from .models.test_generic_model import Model1


class TestExport(unittest.TestCase):
  def test_export(self):
    export_path = tempfile.mkdtemp()
    model = Model1('mix_type2', factor=4, n_layers_before_tf=1).model
    export(model, export_path, {'export_strategy_cloud': ""})

    image = np.ones((1, 224, 224, 3), dtype=np.float32)/127.5 - 1.

    # Then import to check if it works
    # tf.keras.models.load_model also works as working with keras, but useless because we don't need keras data
    model2 = tf.saved_model.load(export_path)

    output1 = model(image)
    output2 = model2(image)
    self.assertTrue(np.allclose(output1.numpy(), output2.numpy()))

    # model3 = tf.keras.models.load_model(export_path)
    # model3.summary()
    # for i in range(7):
    #     print(model.get_layer(index=i))

    shutil.rmtree(export_path)
