import onnx
import typing as t
from tensorflow import keras
from tensorflow.keras import activations

from .common import converter


@converter('Sigmoid')
def sigmoid(node: onnx.NodeProto, input_layer, *args):
    """
    Convert the add operation
    Args:
        node: The node that we wish to convert
    Returns:
    """
    return keras.layers.Activation(activations.sigmoid)(input_layer)


@converter('Relu')
def relu(node: onnx.NodeProto, input_layer, *args):
    """
    Convert the add operation
    Args:
        node: The node that we wish to convert
    Returns:
    """
    return keras.layers.Activation(activations.relu)(input_layer)
