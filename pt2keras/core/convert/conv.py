"""
All convolution operation converter
"""
import torch.nn as nn
from tensorflow import keras

from .common import converter


@converter(nn.Conv2d, keras.layers.Conv2D)
def conv2d(pytorch_conv2d: nn.Conv2d) -> keras.layers.Layer:
    """
    Given a PyTorch conv2d layer, output the equivalent keras conversion
    Args:
        pytorch_conv2d: The conv2d layer to convert
        keras_conv2d: The output conv2d layer in keras

    Returns:
        The converted conv2d layer
    """

    # in_channels = pytorch_conv2d.in_channels
    out_channels = pytorch_conv2d.out_channels

    weights = []
    if pytorch_conv2d.weight is not None:
        weights.append(pytorch_conv2d.weight.data.numpy().transpose((2, 3, 1, 0)))
    if pytorch_conv2d.bias is not None:
        weights.append(pytorch_conv2d.bias.data.numpy())

    # Add Stride
    strides = pytorch_conv2d.stride
    keras_layer = keras.layers.Conv2D(
        out_channels,
        pytorch_conv2d.kernel_size,
        strides=strides,
        weights=weights,
    )
    return keras_layer


