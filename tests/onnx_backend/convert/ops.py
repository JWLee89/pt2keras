from copy import deepcopy

import numpy as np
import pytest
import tensorflow as tf
import torch

from src.pt2keras.onnx_backend.testing.utils import is_approximately_equal

np.set_printoptions(formatter={'float': lambda x: f'{x:0.9f}'})


@pytest.mark.parametrize('input_shape', [(1, 3, 32, 32)])
@pytest.mark.parametrize('op', [torch.abs, torch.acos, torch.ceil])
def test_op(converter, input_shape, op):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.op = op

        def forward(self, x):
            return self.op(x)

    pytorch_model = Model().eval()
    keras_model = converter.convert(pytorch_model, input_shape)
    x_pt = torch.rand(input_shape)
    # Generate dummy inputs
    x_keras = tf.convert_to_tensor(deepcopy(x_pt.numpy()))

    # input dimensions for PyTorch are BCHW, whereas TF / Keras default is BHWC
    if len(x_keras.shape) == 4:
        x_keras = tf.transpose(x_keras, (0, 2, 3, 1))

    output_keras = keras_model(x_keras)
    output_pt = pytorch_model(x_pt)
    is_approximately_equal(output_pt.detach().cpu().numpy(), output_keras.numpy(), atol=1e-4, strict=True)


@pytest.mark.skip('does not support multi-input yet')
@pytest.mark.parametrize('input_shape', [(1, 3, 32, 32)])
@pytest.mark.parametrize(
    'op',
    [
        torch.add,
    ],
)
def test_two_input_ops(converter, input_shape, op):
    class Model(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, lhs, rhs):
            return self.op(lhs, rhs)

    pytorch_model = Model(op).eval()
    a = torch.randn(input_shape)
    b = torch.randn(input_shape)
    keras_model = converter.convert(pytorch_model, (input_shape, input_shape))
    # Generate dummy inputs
    a_keras = tf.convert_to_tensor(deepcopy(a.numpy()))
    b_keras = tf.convert_to_tensor(deepcopy(b.numpy()))

    # input dimensions for PyTorch are BCHW, whereas TF / Keras default is BHWC
    if len(a_keras.shape) == 4:
        a_keras = tf.transpose(a_keras, (0, 2, 3, 1))
        b_keras = tf.transpose(b_keras, (0, 2, 3, 1))

    output_keras = keras_model(a_keras, b_keras)
    output_pt = pytorch_model(a, b)
    is_approximately_equal(output_pt.detach().cpu().numpy(), output_keras.numpy(), atol=1e-4, strict=True)
