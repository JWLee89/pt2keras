import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import onnx
from onnx import numpy_helper
#
# from pt2keras import Pt2Keras
# import tensorflow as tf
#
# from pt2keras.core.convert.common import converter
# from pt2keras.core.models import EfficientNet


conv = nn.Conv2d(3, 16, (1, 1), (2, 2), bias=True)
act = nn.SiLU()
conv_transpose = nn.ConvTranspose2d(16, 32, kernel_size=(2, 2), stride=(2, 2))


#

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # These can all be found using named_modules() or children()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, (1, 1), (2, 2), bias=True),
            nn.BatchNorm2d(16),
        )
        self.pool = nn.Sequential(
                nn.MaxPool2d(2),
                nn.ConvTranspose2d(16, 32, kernel_size=(2, 2), stride=(2, 2)),
                # nn.Sequential(
                #     nn.Flatten(1)
                # )
            )
        self.act = nn.SiLU()

    def forward(self, X):
        # we can retrieve these operations via .modules(), named_modules(), children(), etc.
        output = self.conv(X)
        output = self.act(output)
        output = self.pool(output)

        # But not this
        output = torch.flatten(output)

        # Or this
        final_output = torch.sigmoid(output)
        final_output += 10
        return final_output


def analyze(model):
    for name, module in model.named_children():
        if not list(module.children()) == []:  # if not leaf node, ski[
            print(f'has children: {name}')
        print(f'Name: {name}, module: {module}')


if __name__ == '__main__':
    model = Model()

    analyze(model)
    x = torch.ones(1, 3, 32, 32)

    import torch._C as _C
    TrainingMode = _C._onnx.TrainingMode

    # torch.onnx.export(model,
    #                   x,
    #                   'test_model.onnx',
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   verbose=False,
    #                   training=TrainingMode.TRAINING,
    #                   export_params=True,
    #                   input_names=['input_0'],
    #                   output_names=['output_0'])
    #
    # torch.onnx.export(model,
    #                   x,
    #                   'test_model.onnx',
    #                   do_constant_folding=True,  # whether to execute constant folding for optimization
    #                   export_params=True,
    #                   input_names=['input_0'],
    #                   output_names=['output_0'])
    #
    # # Load onnx model
    onnx_model = onnx.load_model('test_model.onnx')
    graph = onnx_model.graph

    # for init in graph.initializer:
    #     print(init)

    print('-' * 100)
    nodes = graph.node
    node_count = 0

    float_type = onnx.TensorProto.DataType.FLOAT

    for node in nodes:
        print(f'Node: {node}')
        print(f'Node name: {node.name}, '
              f'Op type: {node.op_type}, '
              f'input: {node.input}, '
              f'output: {node.output}, ')
        # node inputs
        for idx, node_input_name in enumerate(node.input):
            print(f'input node: {idx}, {node_input_name}')
        # node outputs
        for idx, node_output_name in enumerate(node.output):
            print(f'Output node: {idx}, {node_output_name}')

        print('*' * 50)
        node_count += 1

    weights = onnx_model.graph.initializer
    for weight in weights:
        np_weights = numpy_helper.to_array(weight)
        print(f'Weights: {np_weights.shape}')

    print(f'Node count: {node_count}')

    inputs = graph.input


    # Get input shape
    for graph_input in inputs:
        input_shape = []
        for d in graph_input.type.tensor_type.shape.dim:
            if d.dim_value == 0:
                input_shape.append(None)
            else:
                input_shape.append(d.dim_value)
        print(
            f"Input Name: {graph_input.name}, Input Data Type: {graph_input.type.tensor_type.elem_type}, Input Shape: {input_shape}"
        )

#
# if __name__ == '__main__':
#
#
#     converter = Pt2Keras()
#     converter.set_logging_level(logging.DEBUG)
#     model = Model()
#     supported_layers, unsupported_layers = converter.inspect(model)
#     print(f'Supported layers: {supported_layers}')
#     print(f'Unsupported layers: {unsupported_layers}')
#     # print(f'Is convertible: {converter.is_convertible(layers)}')
#
#     x = torch.ones(1, 3, 32, 32)
#     converted_model = converter.convert(model)
#     output_pt = model(x)
#     if len(output_pt.shape) == 4:
#         output_pt = output_pt.permute(0, 2, 3, 1)
#
#     x = tf.ones((1, 32, 32, 3))
#     output = converted_model(x)
#     converted_model.summary()
#
#     print(f'Output: {output_pt.shape}')
#     print(f'Output: {output.shape}')
#     output_pt = output_pt.detach().cpu().numpy()
#
#     # print(output_pt)
#     # print(output)
#
#     # assert np.allclose(output_pt, output, atol=1e-2), 'outputs are different'
