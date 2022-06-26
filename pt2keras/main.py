import logging
import os
import typing as t

import torch.nn as nn

from pt2keras.core.onnx.graph import Graph
from pt2keras.core.paths import get_converter_absolute_path


class Pt2Keras:
    _AVAILABLE_IR = ('onnx',)
    _SUPPORTED_LAYERS = {key: {} for key in _AVAILABLE_IR}
    _LOGGER = logging.getLogger()
    _CONVERTERS_IMPORTED = False

    def __init__(self, opset_version: int = 13):
        self.graph = None
        self.opset_version = opset_version
        logging.basicConfig()

    def _validate(self):
        if self.intermediate_rep not in Pt2Keras._AVAILABLE_IR:
            raise ValueError(
                f'Intermediate representation value - {self.intermediate_rep} '
                f'is not available. Choices: {Pt2Keras._AVAILABLE_IR}'
            )

    def _init_converters(self, model: t.Union[nn.Module, str]):
        """
        Initialize converters when Pt2Keras is initialized.
        Args:
            model: The model that we are aiming to convert

        Returns:

        """
        # check model type
        if isinstance(model, nn.Module) or (isinstance(model, str) and model.endswith('.onnx')):
            self.intermediate_rep = Pt2Keras._AVAILABLE_IR[0]
        else:
            raise ValueError(
                f'Invalid model type. ' f'Please pass in one of the following values: {Pt2Keras._AVAILABLE_IR}'
            )

    @property
    def intermediate_rep(self):
        return self._intermediate_rep

    @intermediate_rep.setter
    def intermediate_rep(self, value):
        # Import all converters
        if value not in Pt2Keras._AVAILABLE_IR:
            raise ValueError(f'Invalid intermediate rep: {value}')
        self.graph = Graph(opset_version=self.opset_version)
        self._intermediate_rep = value

    @staticmethod
    def _get_key(layer: t.ClassVar):
        try:
            return layer.__name__
        except Exception:
            return layer.__class__.__name__

    def set_logging_level(self, logging_level):
        Pt2Keras._LOGGER.setLevel(logging_level)

    def convert(self, model, input_shape):
        """
        Perform conversion
        Args:
            model:
            input_shape:

        Returns:

        """
        self._init_converters(model)
        self._validate()
        return self.graph.convert(model, input_shape)

    def inspect(self, model: nn.Module) -> t.Tuple[t.List, t.List]:
        """
        Given a PyTorch model, return a list of
        unique modules in the model.
        """
        return self.graph.inspect(model)

    def is_convertible(self, model):
        supported_layers, unsupported_layers = self.inspect(model)
        return len(unsupported_layers) == 0

    def convert_layer(self, layer: nn.Module):
        return self.graph.convert_layeR(layer)


# Import converters during import
def import_converters():
    if not Pt2Keras._CONVERTERS_IMPORTED:
        converter_directory_path = get_converter_absolute_path()
        for entry in os.scandir(converter_directory_path):
            if entry.is_file():
                # remove '.py' from import statement
                string = f'from pt2keras.core.onnx.convert import {entry.name}'[:-3]
                exec(string)
        Pt2Keras._CONVERTERS_IMPORTED = True


import_converters()
