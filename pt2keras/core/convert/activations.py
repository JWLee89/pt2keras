import tensorflow as tf
import torch.nn as nn
from tensorflow import keras

from .common import converter


@converter(nn.SiLU, keras.activations.swish)
def silu(pytorch_layer):
    """
        Given a PyTorch conv2d layer, output the equivalent keras conversion
        Args:
            pytorch_conv2d: The conv2d layer to convert

        Returns:
            The converted conv2d layer
        """
    # Add Stride
    keras_layer = tf.keras.layers.Lambda(keras.activations.swish)
    return keras_layer
