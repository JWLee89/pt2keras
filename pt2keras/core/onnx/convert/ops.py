import numpy as np
import onnx
import typing as t
from tensorflow import keras

from .common import converter


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
    outputs = current_inputs + computational_graph[node.input_nodes[-1]]
    return outputs


@converter('Mul')
def multiply(node: onnx.NodeProto, computational_graph, current_inputs):
    outputs = current_inputs * computational_graph[node.input_nodes[-1]]
    return outputs


@converter('Div')
def multiply(node: onnx.NodeProto, computational_graph, current_inputs):
    try:
        outputs = current_inputs / computational_graph[node.input_nodes[-1]]
    except:
        # Bracket divide
        outputs = current_inputs / node.weights[0]
    return outputs
