"""
Converter modules will be located here.

TODO:

1. Create inspect module to scan for decorators decorated with
converter


"""
import logging

import torch.nn as nn


class PtKeras:
    """
    Converter for converting PyTorch model to Keras model.

    """
    def __init__(self):
        self.supported_layer = {}
        self.logger = logging.getLogger(__name__)

    def convert(self, pytorch_model: nn.Module) -> None:
        if not isinstance(pytorch_model, nn.Module):
            raise ValueError('Please pass in a PyTorch model')

    def _convert_layer(self, pytorch_layer: nn.Module) -> None:
        if not isinstance(pytorch_layer, nn.Module):
            raise TypeError(f'Not a valid Pytorch layer. Passed in {type(pytorch_layer)}')

    def inspect(self, pytorch_model: nn.Module):

        for name, module in pytorch_model.named_children():
            module_class = module.__class__
            # if module_class not in self.supported_layers:
            #     self.logger.warning(f'{module_class} is not supported. '
            #                         f'Please add the converter')
