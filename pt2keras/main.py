import logging
import typing as t

import torch.nn as nn
from tensorflow import keras


class Pt2Keras:

    _SUPPORTED_LAYERS = {}
    _LOGGER = logging.getLogger()

    _LAYER_KEY = '__class__'

    def __init__(self):
        logging.basicConfig()

    @staticmethod
    def _get_key(layer: t.ClassVar):
        return getattr(layer, Pt2Keras._LAYER_KEY)

    def set_logging_level(self, logging_level):
        Pt2Keras._LOGGER.setLevel(logging_level)

    def convert(self, model: nn.Module):
        return self._convert(model)

    def convert_layer(self, layer: nn.Module):
        key = Pt2Keras._get_key(layer)
        if key in Pt2Keras._SUPPORTED_LAYERS:
            Pt2Keras._LOGGER.debug(f'Converting {layer} ... ')
            conversion_function = Pt2Keras._SUPPORTED_LAYERS[key]
            keras_layer = conversion_function(layer)
            return keras_layer
        else:
            raise ValueError(f'Layer: {layer.__class__.__name__} is not supported ... ')

    def _convert(self, model: nn.Module):
        count = 0
        keras_model = keras.Sequential()
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                Pt2Keras._LOGGER.info('Skipping nn.Sequential ... ')
                continue
            keras_layer = self.convert_layer(module)
            keras_model.add(keras_layer)
            count += 1

        Pt2Keras._LOGGER.debug(f'Processed: {count} layers')
        Pt2Keras._LOGGER.debug(f'Registered converters: {Pt2Keras._SUPPORTED_LAYERS}')
        return keras_model

