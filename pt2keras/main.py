import logging
import typing as t

import torch.nn as nn
from tensorflow import keras


class Pt2Keras:

    _SUPPORTED_LAYERS = {}
    _LOGGER = logging.getLogger()

    _LAYER_KEY = '__name__'

    def __init__(self):
        logging.basicConfig()

    @staticmethod
    def _get_key(layer: t.ClassVar):
        # return getattr(layer, Pt2Keras._LAYER_KEY)
        try:
            return layer.__name__
        except:
            return layer.__class__.__name__

    def set_logging_level(self, logging_level):
        Pt2Keras._LOGGER.setLevel(logging_level)

    def convert(self, model: nn.Module, output_data):
        return self._convert(model, output_data)

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
            key = Pt2Keras._get_key(module)
            if key in Pt2Keras._SUPPORTED_LAYERS and key not in supported_layers:
                supported_layers.append(key)
            # We dont count sequential as a unique layer
            elif key not in Pt2Keras._SUPPORTED_LAYERS and key not in unsupported_layers:
                unsupported_layers.append(key)

        return supported_layers, unsupported_layers

    def is_convertible(self, model):
        supported_layers, unsupported_layers = self.inspect(model)
        return len(unsupported_layers) == 0

    def convert_layer(self, layer: nn.Module):
        key = Pt2Keras._get_key(layer)
        if key in Pt2Keras._SUPPORTED_LAYERS:
            Pt2Keras._LOGGER.debug(f'Converting {layer} ... ')
            conversion_function = Pt2Keras._SUPPORTED_LAYERS[key]
            keras_layer = conversion_function(layer)
            return keras_layer
        else:
            raise ValueError(f'Layer: {layer.__class__.__name__} is not supported ... ')

    def _convert(self, model: nn.Module, output_data):
        count = 0
        keras_model = keras.Sequential()
        items = []
        for name, module in model.named_modules():
            if not list(module.children()) == []:  # if not leaf node, skip
                continue

            if isinstance(module, nn.Sequential):
                Pt2Keras._LOGGER.info('Skipping nn.Sequential ... ')
                continue
            keras_layer = self.convert_layer(module)
            keras_model.add(keras_layer)
            items.append(keras_layer)
            count += 1

        str_repr = '\n'.join(f'{i + 1} - {layer}' for i, layer in enumerate(items))
        Pt2Keras._LOGGER.debug(f'Successfully converted {count} PyTorch layers: '
                               f'\n{str_repr}')
        return keras_model
