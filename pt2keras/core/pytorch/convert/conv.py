"""
All convolution operation converters
"""
import typing as t

import torch.nn as nn
from tensorflow import keras

from .common import converter


@converter(nn.Conv2d)
def conv2d(pytorch_conv2d: nn.Conv2d) -> keras.layers.Layer:
    """
    Given a PyTorch conv2d layer, output the equivalent keras conversion
    Args:
        pytorch_conv2d: The conv2d layer to convert

    Returns:
        The converted conv2d layer
    """

    # in_channels = pytorch_conv2d.in_channels
    out_channels = pytorch_conv2d.out_channels

    weights = []
    if pytorch_conv2d.weight is not None:
        weights.append(pytorch_conv2d.weight.detach().numpy().transpose((2, 3, 1, 0)))
    if pytorch_conv2d.bias is not None:
        weights.append(pytorch_conv2d.bias.detach().numpy())

    # Add Stride
    strides = pytorch_conv2d.stride
    keras_layer = keras.layers.Conv2D(
        out_channels,
        pytorch_conv2d.kernel_size,
        strides=strides,
        weights=weights,
    )
    return keras_layer


@converter(nn.ConvTranspose2d)
def conv2d_transpose(layer: nn.ConvTranspose2d):
    filters = layer.out_channels
    kernel_size: t.Tuple = layer.kernel_size
    strides = layer.stride
    padding = layer.padding
    dilation = layer.dilation

    weights = []
    if layer.weight is not None:
        weights.append(layer.weight.detach().numpy().transpose((2, 3, 1, 0)))
    if layer.bias is not None:
        weights.append(layer.bias.detach().numpy())

    stride_val = strides if isinstance(strides, int) else strides[0]
    dilation = dilation if isinstance(dilation, int) else dilation[0]

    # see: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
    # an integer, specifying the dilation rate for all spatial dimensions for dilated convolution.
    # Specifying different dilation rates for different dimensions is not supported.
    # Currently, specifying any dilation_rate value != 1 is
    # incompatible with specifying any stride value != 1.
    if stride_val != 1 and dilation > 1:
        raise ValueError('In Keras, specifying any dilation_rate value != 1 '
                         'is incompatible with specifying any stride value != 1')

    keras_layer = keras.layers.Conv2DTranspose(filters, kernel_size,
                                               strides=strides,
                                               output_padding=padding,
                                               dilation_rate=dilation,
                                               weights=weights)
    return keras_layer
