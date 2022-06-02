"""
All batch norm operation converters
"""
import typing as t

import torch.nn as nn
from tensorflow import keras

from .common import converter


# @converter(nn.BatchNorm2d)
# def batch_norm_1d(layer: nn.Module):
#
#     keras_layer = keras.layers.BatchNormalization()
#     return keras_layer


@converter(nn.BatchNorm1d)
def batch_norm_2d(layer: nn.Module):
    epsilon = layer.eps
    momentum = layer.momentum
    weights = [layer.weight.data.numpy()]
    weights.append(layer.bias.data.numpy())
    weights.append(layer.running_mean.data.numpy())
    var = layer.running_var
    weights.append(var)

    keras_layer = keras.layers.BatchNormalization(
        axis=-1,
        momentum=momentum,
        epsilon=epsilon,
        weights=weights,
    )
    return keras_layer


@converter(nn.BatchNorm2d)
def batch_norm_2d(layer: nn.Module):
    epsilon = layer.eps
    momentum = layer.momentum
    weights = [layer.weight.data.numpy()]
    weights.append(layer.bias.data.numpy())
    weights.append(layer.running_mean.data.numpy())
    var = layer.running_var
    weights.append(var)

    keras_layer = keras.layers.BatchNormalization(
        axis=-1,
        momentum=momentum,
        epsilon=epsilon,
        weights=weights,
    )
    return keras_layer
