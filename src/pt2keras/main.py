import logging
import typing as t

import torch.nn as nn

from .onnx_backend.graph import Graph

# We need to grab all the converters
exec('from .onnx_backend.convert import *')


class Pt2Keras:
    _AVAILABLE_IR = ('onnx',)
    _SUPPORTED_LAYERS = {key: {} for key in _AVAILABLE_IR}
    _LOGGER = logging.getLogger(__name__)
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

    def convert(self, model, input_shape, strict: bool = False):
        """
        Perform conversion
        Args:
            model: The PyTorch / onnx model to convert
            input_shape: The input Tensor shape
            strict: If set to true, an error will be thrown if
            any single tensor value deviates by a certain threshold.

        Returns:
            The converted keras version of the PyTorch model
        """
        self._init_converters(model)
        self._validate()
        return self.graph.convert(model, input_shape, strict)

    def inspect(self, model: nn.Module) -> t.Tuple[t.List, t.List]:
        """
        Given a PyTorch model, return a list of
        unique modules in the model.
        """
        return self.graph.inspect(model)

    def is_convertible(self, model):
        supported_layers, unsupported_layers = self.inspect(model)
        return len(unsupported_layers) == 0
