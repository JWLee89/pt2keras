"""
Store all PyTorch -> Keras converters ehre
"""
import typing as t

import torch.nn as nn
from tensorflow import keras


_SUPPORTED_LAYERS = {}


def _extract_parameters() -> t.Tuple:
    """
    Extract parameter
    """
    pass


def converter(pytorch_module: nn.Module, keras_equivalent: keras.layers.Layer) -> t.Callable:
    """
    Decorator for adding custom converters.
    This will inspect all functions decorated with
    converters
    """
    if not isinstance(pytorch_module, nn.Module):
        raise ValueError(f'Please pass in a nn.Module. Passed in: {pytorch_module}')

    if not isinstance(keras_equivalent, keras.layers.Layer):
        raise ValueError(f'Please pass in a keras.layers.Layer. Passed in: {keras_equivalent}')

    def inner(wrapped_fn: t.Callable) -> t.Callable:
        def created_converter(pytorch_layer: nn.Module, *args, **kwargs) -> t.Any:
            """
            Args:
                pytorch_layer: The PyTorch layer that should be converted into Keras
                *args:
                **kwargs:

            Returns:

            """
            if not isinstance(pytorch_layer, nn.Conv2d):
                raise TypeError(f'{type(pytorch_layer)} is not a valid PyTorch layer')

            output = wrapped_fn(pytorch_layer, *args, **kwargs)

            return output
        return created_converter

    return inner
