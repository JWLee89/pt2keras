import onnx
import typing as t
from tensorflow import keras
from tensorflow.keras import backend as K

from .common import converter
from ..util import tensor_proto_to_tf_dtype


@converter('Conv')
def add(node: onnx.NodeProto, computational_graph, current_inputs):
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
    )(current_inputs)
    return outputs


@converter('Constant')
def add(node: onnx.NodeProto, computational_graph, current_inputs):
    """
    Convert the add operation
    Args:
        node: The node that we wish to convert
    Returns:
    """
    return current_inputs


@converter('Add')
def add(node: onnx.NodeProto, computational_graph, current_inputs):
    outputs = computational_graph[node.input_nodes[0]] + computational_graph[node.input_nodes[-1]]
    return outputs


@converter('Mul')
def multiply(node: onnx.NodeProto, computational_graph, current_inputs):
    outputs = computational_graph[node.input_nodes[0]] * computational_graph[node.input_nodes[-1]]
    return outputs


@converter('Div')
def divide(node: onnx.NodeProto, computational_graph, current_inputs):
    outputs = computational_graph[node.input_nodes[0]] / computational_graph[node.input_nodes[-1]]
    return outputs


@converter('Cast')
def cast(node: onnx.NodeProto, computational_graph, current_inputs):
    """
    Floor divide is considered a Cast operation in onnx
    Args:
        node:
        computational_graph:
        current_inputs:

    Returns:

    """
    print(f'computational_graph: {computational_graph}')
    tf_dtype = tensor_proto_to_tf_dtype(node.attributes['to'])
    outputs = K.cast(current_inputs, dtype=tf_dtype)
    return outputs
