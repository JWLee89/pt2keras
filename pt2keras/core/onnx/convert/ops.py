import onnx
import typing as t
from tensorflow import keras
from tensorflow.keras import backend as K

from .common import converter
from ..util import tensor_proto_to_tf_dtype


@converter('Conv')
def add(node: onnx.NodeProto, input_layer, *node_inputs):
    """
    Convert the add operation
    Args:
        node: The node that we wish to convert
    Returns:

    """
    attributes: t.Dict = node.attributes
    weights_shape = node.weights[0].shape
    filter_count = weights_shape[-1]
    outputs = keras.layers.Conv2D(
        filter_count,  # filters
        attributes['kernel_shape'],  # Kernel size
        strides=attributes['strides'],
        groups=attributes['group'],
        weights=node.weights,
        dilation_rate=attributes['dilations'],
        # Weights is of length two ['weights', 'bias']
        use_bias=len(node.weights) == 2,
    )(input_layer)
    return outputs


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
    Floor divide is considered a Cast operation in onnx
    Args:
        node:
        computational_graph:
        current_inputs:

    Returns:

    """
    tf_dtype = tensor_proto_to_tf_dtype(node.attributes['to'])
    outputs = K.cast(input_layer, dtype=tf_dtype)
    return outputs
