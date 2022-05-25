from . import *


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
        weights.append(pytorch_layer.weight.data.numpy().transpose((2, 3, 1, 0)))
    if pytorch_layer.bias is not None:
        weights.append(pytorch_layer.bias.data.numpy())
    if weights:
        keras_layer.set_weights(weights)


_SUPPORTED_LAYERS = {}


def converter(pytorch_module: nn.Module, keras_equivalent) -> t.Callable:
    """
    Decorator for adding custom converters.
    This will inspect all functions decorated with
    converters
    """
    if not isinstance(pytorch_module, nn.Module):
        raise ValueError(f'Please pass in a nn.Module. Passed in: {pytorch_module}')

    if pytorch_module.__class__.__name__ in _SUPPORTED_LAYERS:
        raise DuplicateLayerConverterError(f'{pytorch_module.__class__.__name__} converter already exists ...')

    _LOGGER.debug(f'Registering pytorch layer: {pytorch_module.__class__}')

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

            # Should add all available arguments and so on
            keras_layer = wrapped_fn(pytorch_layer, keras_equivalent, *args, **kwargs)

            # Post processing
            # -------------------

            # 1. Add weights and bias
            _add_weights_and_bias_to_keras(pytorch_layer, keras_layer)

            return keras_layer
        return created_converter

    return inner

