import onnx
import typing as t
from tensorflow import keras
from tensorflow.keras import activations

from .common import converter


@converter('Sigmoid')
def add(node: onnx.NodeProto, computational_graph, current_inputs):
    """
    Convert the add operation
    Args:
        node: The node that we wish to convert
    Returns:
    """
    return keras.layers.Activation(activations.sigmoid)(current_inputs)
