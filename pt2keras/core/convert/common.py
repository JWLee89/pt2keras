import typing as t

import numpy as np
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


def int_to_2d_tuple(num: int, size: int = 2):
    """
    Convert int to tuple.
    This is for operations that require a tuple instead of an int.
    Args:
        num:
        size:

    Returns:

    """
    if not isinstance(num, int):
        if not isinstance(num, (t.Tuple, t.List)):
            raise ValueError('Num must be a tuple or list')
    else:
        return tuple([num] * size)


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
    if isinstance(pt_layer, torch.nn.modules.conv._ConvNd):
        assert pt_layer.kernel_size[0] == pt_layer.kernel_size[1], \
            f'kernel_size should be square. Actual: {pt_layer.kernel_size}'
        x_pt = torch.randn(batch_size, pt_layer.in_channels, height, width)
    elif isinstance(pt_layer, nn.Linear):
        x_pt = torch.randn(batch_size, pt_layer.in_features)
    else:
        x_pt = torch.randn(batch_size, channel, height, width)

    # (B, C, H, W) -> (B, H, W, C)
    x_keras = tf.convert_to_tensor(x_pt.data.numpy().transpose(0, 2, 3, 1))
    return x_pt, x_keras


def _test_layer(pt_layer: nn.Module, keras_layer, batch_size: int = 2, atol=1e-4):
    """
    Test whether the output of a
    Args:
        pt_layer:
        keras_layer:
        batch_size:

    Returns:

    """

    x_pt, x_keras = get_test_input_data(pt_layer)
    # get Pt output
    output_pt = pt_layer(x_pt)
    # A batch of images
    if len(output_pt == 4):
        # Change PyTorch dimension format to Keras
        output_pt = output_pt.permute(0, 2, 3, 1)
    output_keras = keras_layer(x_keras)

    # Convert output to numpy
    output_pt = output_pt.detach().numpy()
    output_keras = output_keras.numpy()

    # Average diff over all axis
    average_diff = np.mean(np.mean([output_pt, output_keras]))

    pt_class_name = pt_layer.__class__.__name__
    keras_class_name = keras_layer.__class__.__name__
    header = f'---------------PyTorch {pt_class_name} --> ' \
             f'Keras {keras_class_name}---------------'
    footer = '-' * len(header)
    _LOGGER.debug(f'\n\t{header}'
                  f'\n\tPyTorch Input: {x_pt.shape}.'
                  f'\n\tKeras Input: {output_pt.shape}'
                  f'\n\tPyTorch Output: {output_pt.shape}'
                  f'\n\tKeras Output: {output_keras.shape}'
                  f'\n\tOutput average diff: {average_diff}'
                  f'\n\t{footer}')

    assert output_keras.shape == output_pt.shape, f'expected: {output_keras.shape}, Actual: {output_pt.shape}'
    output_is_approximately_equal = np.allclose(output_pt, output_keras, atol=atol)
    assert output_is_approximately_equal, f'PyTorch output and Keras output is different for layer: {pt_class_name}. ' \
                                          f'Mean difference: {average_diff}'


def converter(pytorch_module: t.ClassVar,
              output_testing_fn: t.Callable = None,
              override: bool = False) -> t.Callable:
    """
    Decorator for adding custom converters.
    This will inspect all functions decorated with converters

    Args:
        pytorch_module: The target PyTorch module to convert (into Keras)
        output_testing_fn (optional): Tests that the output of the PyTorch layer is identical to
        the Keras equivalent.
        override (bool): if set to true, will override and replace any existing converters
    """
    if not issubclass(pytorch_module, nn.Module):
        raise ValueError(f'Please pass in a nn.Module. Passed in: {pytorch_module}')

    # Retrieve unique key
    key = Pt2Keras._get_key(pytorch_module)

    if override:
        _LOGGER.warning(f'WARNING:: Overriding existing converter for PyTorch layer: {pytorch_module}. '
                        f'Please double check to ensure that this is the desired behavior.')

    if not override and key in Pt2Keras._SUPPORTED_LAYERS:
        raise DuplicateLayerConverterError(f'{key} converter already exists ...')

    def inner(wrapped_fn: t.Callable) -> t.Callable:

        @wraps(wrapped_fn)
        def created_converter(pytorch_layer, *args, **kwargs) -> t.Any:
            """
            Given a pytorch operation or layer, directly port it to keras
            Args:
                pytorch_layer: The PyTorch layer that should be converted into Keras
            Returns:
                The converted keras layer with all states copied / patched onto the keras layer
            """

            # Should add all available arguments and so on
            keras_layer = wrapped_fn(pytorch_layer, *args, **kwargs)

            # Post processing
            # -------------------

            # 1. Add weights and bias
            # _add_weights_and_bias_to_keras(pytorch_layer, keras_layer)

            # 2. Perform tests to see whether the two layers (pytorch and keras)
            # are outputting the same value and shape
            test_layer = _test_layer if output_testing_fn is None else output_testing_fn
            test_layer(pytorch_layer, keras_layer, *args, **kwargs)

            return keras_layer

        if not override:
            Pt2Keras._LOGGER.warning(f'Registering pytorch converter for layer: {pytorch_module}')
        Pt2Keras._SUPPORTED_LAYERS[key] = created_converter
        return created_converter

    return inner
