import onnx
import typing as t
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
