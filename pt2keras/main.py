import logging
import os
import typing as t

import onnx
import torch.nn as nn


class Pt2Keras:

    _AVAILABLE_IR = ('onnx', 'pytorch')
    _SUPPORTED_LAYERS = {
        key: {} for key in _AVAILABLE_IR
    }
    _LOGGER = logging.getLogger()

    def __init__(self, model, input_shape: t.Tuple):
        self.graph = None
        self.input_shape = input_shape
        self.model = model
        # check model type
        if isinstance(self.model, nn.Module):
            self.intermediate_rep = Pt2Keras._AVAILABLE_IR[0]
            self.model.eval()
        # onnx file path
        elif isinstance(self.model, str) and self.model.endswith('.onnx'):
            if not os.path.exists(self.model):
                raise IOError(f'Cannot find onnx model at specified path: {self.model}')

            self.intermediate_rep = Pt2Keras._AVAILABLE_IR[0]
        else:
            raise ValueError(f'Invalid model type. '
                             f'Please pass in one of the following values: {Pt2Keras._AVAILABLE_IR}')
        logging.basicConfig()
        self._validate()

    def _validate(self):
        if self.intermediate_rep not in Pt2Keras._AVAILABLE_IR:
            raise ValueError(f'Intermediate representation value - {self.intermediate_rep} '
                             f'is not available. Choices: {Pt2Keras._AVAILABLE_IR}')


    @property
    def intermediate_rep(self):
        return self._intermediate_rep

    @intermediate_rep.setter
    def intermediate_rep(self, value):
        # Import all converters
        if value == 'onnx':
            for entry in os.scandir('pt2keras/core/onnx/convert'):
                if entry.is_file():
                    string = f'from pt2keras.core.onnx.convert import {entry.name}'[:-3]
                    exec(string)
            from pt2keras.core.onnx.graph import Graph
        elif value == 'pytorch':
            for entry in os.scandir('pt2keras/core/pytorch/convert'):
                if entry.is_file():
                    string = f'from pt2keras.core.pytorch.convert import {entry.name}'[:-3]
                    exec(string)
            from pt2keras.core.pytorch.graph import Graph
        else:
            raise ValueError('Invalid property')
        self.graph = Graph(self.model, self.input_shape)
        self._intermediate_rep = value

    @staticmethod
    def _get_key(layer: t.ClassVar):
        try:
            return layer.__name__
        except:
            return layer.__class__.__name__

    def set_logging_level(self, logging_level):
        Pt2Keras._LOGGER.setLevel(logging_level)

    def convert(self):
        return self.graph._convert()

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
