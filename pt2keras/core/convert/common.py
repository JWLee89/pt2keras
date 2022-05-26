import typing as t

import tensorflow as tf
import torch
import torch.nn as nn
from functools import wraps
from tensorflow import keras

from ...main import Pt2Keras


class DuplicateLayerConverterError(ValueError):
    pass


import logging

_LOGGER = logging.getLogger(__name__)


def _extract_parameters() -> t.Tuple:
    """
    Extract parameter
    """
    pass


def _add_weights_and_bias_to_keras(pytorch_layer: nn.Module, keras_layer: keras.layers.Layer):
    # Update weights and bias
    weights = []
    if pytorch_layer.weight is not None:
        keras_layer.get_weights()
        weights.append(pytorch_layer.weight.data.numpy().transpose((2, 3, 1, 0)))
    if pytorch_layer.bias is not None:
        weights.append(pytorch_layer.bias.data.numpy())
    if weights:
        keras_layer.set_weights(weights)


def get_test_input_data(pt_layer: nn.Module,
                        batch_size: int = 2,
                        channel: int = 3,
                        height: int = 16,
                        width: int = 16) -> t.Tuple[torch.Tensor, tf.Tensor]:
    if isinstance(pt_layer, nn.Conv2d):
        assert pt_layer.kernel_size[0] == pt_layer.kernel_size[1], \
            f'kernel_size should be square. Actual: {pt_layer.kernel_size}'
        x_pt = torch.randn(batch_size, pt_layer.kernel_size[0], height, width)
    elif isinstance(pt_layer, nn.Linear):
        x_pt = torch.randn(batch_size, pt_layer.in_features)
    else:
        x_pt = torch.randn(batch_size, channel, height, width)

    # (B, C, H, W) -> (B, H, W, C)
    x_keras = tf.convert_to_tensor(x_pt.data.numpy().transpose(0, 2, 3, 1))
    return x_pt, x_keras


def _test_layer(pt_layer: nn.Module, keras_layer, batch_size: int = 2):

    x_pt, x_keras = get_test_input_data(pt_layer)

    # get Pt output
    output_pt = pt_layer(x_pt)
    # A batch of images
    if len(output_pt == 4):
        output_pt = output_pt.permute(0, 2, 3, 1)


def converter(pytorch_module: t.ClassVar) -> t.Callable:
    """
    Decorator for adding custom converters.
    This will inspect all functions decorated with
    converters
    """
    if not issubclass(pytorch_module, nn.Module):
        raise ValueError(f'Please pass in a nn.Module. Passed in: {pytorch_module}')

    key = Pt2Keras._get_key(pytorch_module)

    if key in Pt2Keras._SUPPORTED_LAYERS:
        raise DuplicateLayerConverterError(f'{key} converter already exists ...')

    def inner(wrapped_fn: t.Callable) -> t.Callable:

        @wraps(wrapped_fn)
        def created_converter(*args, **kwargs) -> t.Any:
            """
            Given a pytorch operation or layer, directly port it to keras
            Args:
                pytorch_layer: The PyTorch layer that should be converted into Keras
            Returns:
                The converted keras layer with all states copied / patched onto the keras layer
            """

            # Should add all available arguments and so on
            keras_layer = wrapped_fn(*args, **kwargs)

            # Post processing
            # -------------------

            # 1. Add weights and bias
            # _add_weights_and_bias_to_keras(pytorch_layer, keras_layer)

            return keras_layer

        print(f'Registering pytorch converter for layer: {pytorch_module}')
        Pt2Keras._LOGGER.warning(f'Registering pytorch converter for layer: {pytorch_module}')
        Pt2Keras._SUPPORTED_LAYERS[pytorch_module] = created_converter

        print(Pt2Keras._SUPPORTED_LAYERS)

        return created_converter

    return inner
