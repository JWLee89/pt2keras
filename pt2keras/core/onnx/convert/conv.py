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

    pads = attributes['pads'] if 'pads' in attributes else [0, 0, 0]

    padding = None
    if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
        padding = (pads[0], pads[1])
    elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
        padding = ((pads[0], pads[2]), (pads[1], pads[3]))

    if padding:
        padding_name = node.name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(
            padding=padding,
            name=padding_name,
        )
        input_layer = padding_layer(input_layer)

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


@converter('ConvTranspose')
def add(node: onnx.NodeProto, input_layer, *node_inputs):
    """
    Convert the add operation
    Args:
        node: The node that we wish to convert
    Returns:

    """
    attributes: t.Dict = node.attributes
    weights_shape = node.weights[0].shape
    filter_count = weights_shape[-2]
    padding = 'same' if attributes['pads'][0] != 0 \
                        and attributes['pads'][1] != 0 \
                        and attributes['pads'][1] == attributes['pads'][0] else 'valid'

    outputs = keras.layers.Conv2DTranspose(
        filter_count,  # filters
        attributes['kernel_shape'],  # Kernel size
        strides=attributes['strides'],
        groups=attributes['group'],
        padding=padding,
        output_padding=attributes['pads'][:2],
        weights=node.weights,
        dilation_rate=attributes['dilations'],
        # Weights is of length two ['weights', 'bias']
        use_bias=len(node.weights) == 2,
    )(input_layer)
    return outputs
