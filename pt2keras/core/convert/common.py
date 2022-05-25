import typing as t

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


def converter(pytorch_module: t.ClassVar, keras_equivalent) -> t.Callable:
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
