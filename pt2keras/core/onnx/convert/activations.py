import onnx
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
    activation_layer = keras.layers.Activation(activations.sigmoid)
    return activation_layer(input_layer), activation_layer


@converter('Relu')
def relu(node: onnx.NodeProto, input_layer, *args):
    """
    Convert the add operation
    Args:
        node: The node that we wish to convert
    Returns:
    """
    activation_layer = keras.layers.Activation(activations.relu)
    return activation_layer(input_layer), activation_layer


@converter('HardSigmoid')
def hard_sigmoid(node: onnx.NodeProto, input_layer, *args):
    activation_layer = keras.layers.Activation(activations.hard_sigmoid)
    return activation_layer(input_layer), activation_layer
