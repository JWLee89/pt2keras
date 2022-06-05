import onnx
import typing as t

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from .common import converter
from ..util import tensor_proto_to_tf_dtype


@converter('Constant')
def constant(node: onnx.NodeProto, input_layer, *inputs):
    """
    A operation that outputs the input
    Args:
        node: The node that we wish to convert
    Returns:
    """
    return input_layer


@converter('Add')
def add(node: onnx.NodeProto, input_layer, lhs, rhs):
    return lhs + rhs


@converter('Mul')
def multiply(node: onnx.NodeProto, input_layer, lhs, rhs):
    return lhs * rhs


@converter('Div')
def divide(node: onnx.NodeProto, input_layer, lhs, rhs):
    return lhs / rhs


@converter('Cast')
def cast(node: onnx.NodeProto, input_layer, *args):
    """
    Floor divide is considered a Cast operation in onnx,
    since we are casting from float32 to int
    """
    tf_dtype = tensor_proto_to_tf_dtype(node.attributes['to'])
    outputs = K.cast(input_layer, dtype=tf_dtype)
    return outputs


@converter('Gather')
def gather(node: onnx.NodeProto, input_layer, input_tensor, indices):
    # the axis to slice across
    axis = node.attributes['axis']
    print(f'shape: {len(input_tensor.shape)}, axis: {axis}')
    # Mapping PyTorch channels to keras
    if len(input_tensor.shape) > 2:
        axis_mapper = {
            0: 0,
            1: 3,
            2: 1,
            3: 2,
        }
        mapped_axis = axis_mapper[axis]
    else:
        mapped_axis = axis
    return tf.gather(input_tensor, indices=indices, axis=mapped_axis)


@converter('Dropout')
def gather(node: onnx.NodeProto, input_layer, input_tensor):
    # TODO: Dropout removed during evaluation phase
    print(f'Node attr: {node.attributes}')
    return keras.layers.Dropout()(input_layer)


@converter('Flatten')
def flatten(node: onnx.NodeProto, input_layer, input_tensor):
    print(f'Flatten: {node.attributes}')
    return keras.layers.Flatten()(input_layer)
