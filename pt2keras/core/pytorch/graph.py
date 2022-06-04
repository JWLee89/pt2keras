
import logging
import typing as t

import torch.nn as nn
from tensorflow import keras


class Graph:

    _LOGGER = logging.getLogger('pytorch::Graph')
    _SUPPORTED_LAYERS = {}

    def __init__(self, model: nn.Module):
        self.model = model
        logging.basicConfig()

    @staticmethod
    def _get_key(layer: t.ClassVar):
        try:
            return layer.__name__
        except:
            return layer.__class__.__name__

    def convert(self):
        return self._convert()

    def inspect(self, model: nn.Module) -> t.Tuple[t.List, t.List]:
        """
        Given a PyTorch model, return a list of
        unique modules in the model.
        """
        supported_layers = []
        unsupported_layers = []
        for name, module in model.named_modules():
            if not list(module.children()) == []:  # if not leaf node, ski[
                continue
            key = Graph._get_key(module)
            if key in Graph._SUPPORTED_LAYERS and key not in supported_layers:
                supported_layers.append(key)
            # We dont count sequential as a unique layer
            elif key not in Graph._SUPPORTED_LAYERS and key not in unsupported_layers:
                unsupported_layers.append(key)

        return supported_layers, unsupported_layers

    def is_convertible(self):
        supported_layers, unsupported_layers = self.inspect(self.model)
        return len(unsupported_layers) == 0

    def convert_layer(self, layer: nn.Module):
        key = Graph._get_key(layer)
        if key in Graph._SUPPORTED_LAYERS:
            Graph._LOGGER.debug(f'Converting {layer} ... ')
            conversion_function = Graph._SUPPORTED_LAYERS[key]
            keras_layer = conversion_function(layer)
            return keras_layer
        else:
            raise ValueError(f'Layer: {layer.__class__.__name__} is not supported ... ')

    def _convert(self):
        count = 0
        keras_model = keras.Sequential()
        items = []

        # We are iterating over PyTorch modules with named parameters
        for name, module in self.model.named_modules():
            if not list(module.children()) == []:  # if not leaf node, skip
                continue

            if isinstance(module, nn.Sequential):
                Graph._LOGGER.info('Skipping nn.Sequential ... ')
                continue
            keras_layer = self.convert_layer(module)
            keras_model.add(keras_layer)
            items.append(keras_layer)
            count += 1

        # Now, we will iterate over onnx computational graph and find those Pesky operations
        str_repr = '\n'.join(f'{i + 1} - {layer}' for i, layer in enumerate(items))
        Graph._LOGGER.debug(f'Successfully converted {count} PyTorch layers: '
                               f'\n{str_repr}')
        return keras_model
