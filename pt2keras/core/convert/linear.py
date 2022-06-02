"""
All convolution operation converter
"""
import torch.nn as nn
from tensorflow import keras

from .common import converter


@converter(nn.Linear)
def max_pool_2d(layer: nn.Linear):
    weights = [layer.weight.data.numpy()]
    if layer.bias is not None:
        weights.append(layer.bias.data.numpy())

    keras_layer = keras.layers.Dense(
        layer.in_features,
        activation=None,
        use_bias=layer.bias is not None,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        weights=weights,
    )
    return keras_layer


@converter(nn.Flatten)
def flatten(layer: nn.Module):
    keras_layer = keras.layers.Flatten()
    return keras_layer