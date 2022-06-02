import logging

import numpy as np
import torch
import torch.nn as nn

from pt2keras import Pt2Keras
import tensorflow as tf

from pt2keras.core.convert.common import converter
from pt2keras.core.models import EfficientNet


conv = nn.Conv2d(3, 16, (1, 1), (2, 2), bias=True)
act = nn.SiLU()
conv_transpose = nn.ConvTranspose2d(16, 32, kernel_size=(2, 2), stride=(2, 2))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            conv,
            nn.BatchNorm2d(16),
            act,
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(2),
            conv_transpose,
            nn.Sequential(
                nn.Flatten(1)
            )
        )

    def forward(self, X):
        """

        Args:
            X:

        Returns:

        """
        # comment
        output = self.conv(X)
        final = self.pool(output)
        return final


class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        return self.model(X)


# import inspect
# from itertools import dropwhile
#
#
# def get_function_body(func):
#     source_lines = inspect.getsourcelines(func)[0]
#     source_lines = dropwhile(lambda x: x.startswith('@'), source_lines)
#     def_line = next(source_lines).strip()
#     if def_line.startswith('def ') and def_line.endswith(':'):
#         # Handle functions that are not one-liners
#         first_line = next(source_lines)
#         # Find the indentation of the first line
#         indentation = len(first_line) - len(first_line.lstrip())
#         return ''.join([first_line[indentation:]] + [line[indentation:] for line in source_lines])
#     else:
#         # Handle single line functions
#         return def_line.rsplit(':')[-1].strip()
#
# import inspect
# filtered_obj = get_function_body(model_to_convert.forward)
# print(filtered_obj)
#
# torch.onnx.export(model_to_convert,
#                   x,
#                   'yee.onnx',
#                   export_params=True,
#                   training=torch.onnx.TrainingMode.EVAL,
#                   )


if __name__ == '__main__':


    converter = Pt2Keras()
    converter.set_logging_level(logging.DEBUG)
    model = Model()
    supported_layers, unsupported_layers = converter.inspect(model)
    print(f'Supported layers: {supported_layers}')
    print(f'Unsupported layers: {unsupported_layers}')
    # print(f'Is convertible: {converter.is_convertible(layers)}')

    x = torch.ones(1, 3, 32, 32)
    converted_model = converter.convert(model, x)
    output_pt = model(x)
    if len(output_pt.shape) == 4:
        output_pt = output_pt.permute(0, 2, 3, 1)

    x = tf.ones((1, 32, 32, 3))
    output = converted_model(x)
    converted_model.summary()

    print(f'Output: {output_pt.shape}')
    print(f'Output: {output.shape}')
    output_pt = output_pt.detach().cpu().numpy()

    # print(output_pt)
    # print(output)

    # assert np.allclose(output_pt, output, atol=1e-2), 'outputs are different'
