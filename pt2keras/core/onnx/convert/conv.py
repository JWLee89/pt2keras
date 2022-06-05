import onnx
import typing as t
from tensorflow import keras

from .common import converter


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



