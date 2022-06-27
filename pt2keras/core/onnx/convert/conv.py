import logging
import typing as t

from tensorflow import keras

from ..graph import OnnxNode
from .common import converter


@converter('Conv')
def conv(node: OnnxNode, input_layer, *inputs):
    """
    Convert the conv operation.
    @credit to onnx2keras where I got the implementation details from:
    Link: https://github.com/gmalivenko/onnx2keras
    Args:
        node: The node that we wish to convert
    """
    logger = logging.getLogger('conv::Conv')
    weights, bias = None, None
    weights = node.weights[0]
    bias = None if len(node.weights) != 2 else node.weights[1]

    # print(f'Conv input: {__.shape}, input shape: {input_layer.shape}, node name: {node.name}')

    attributes: t.Dict = node.attributes
    has_bias = bias is not None
    n_groups = attributes['group'] if 'group' in attributes else 1
    pads = attributes['pads'] if 'pads' in attributes else [0, 0, 0]
    dilation = attributes['dilations'][0] if 'dilations' in attributes else 1
    strides = attributes['strides'] if 'strides' in attributes else [1, 1, 1]

    # Get pads
    padding = None
    if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
        padding = (pads[0], pads[1])
    elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
        padding = ((pads[0], pads[2]), (pads[1], pads[3]))

    # Unlike in PyTorch, we need to manually add a zero-padding layer to get the same behavior.
    # If you use Keras conv2d padding, you will not get the same output dimension as PyTorch padding.
    # This caused me a lot headache before figuring it out thanks to onnx2keras.
    if padding:
        padding_name = node.name + '_pad'
        padding_layer = keras.layers.ZeroPadding2D(padding=padding, name=padding_name, data_format='channels_last')
        input_layer = padding_layer(input_layer)

    weights = weights.transpose(2, 3, 1, 0)
    weights_shape = weights.shape
    height, width, channels_per_group, out_channels = weights_shape
    in_channels = channels_per_group * n_groups

    if n_groups == in_channels and n_groups != 1:
        logger.debug(
            'Number of groups is equal to input channels, use DepthWise convolution. '
            f'Groups: {n_groups}, input channels: {in_channels}'
        )
        weights = weights.transpose(0, 1, 3, 2)

        output_layer = keras.layers.DepthwiseConv2D(
            kernel_size=(height, width),
            strides=(strides[0], strides[1]),
            padding='valid',
            use_bias=has_bias,
            activation=None,
            depth_multiplier=1,
            weights=[weights, bias] if has_bias else [weights],
            dilation_rate=dilation,
            bias_initializer='zeros',
            kernel_initializer='zeros',
        )
        outputs = output_layer(input_layer)
        # # skip test
        output_layer = None

    elif n_groups != 1:
        logger.debug('Number of groups more than 1, but less than number of in_channel, use group convolution')

        # Example from https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
        def target_layer(x, groups=n_groups, stride_y=strides[0], stride_x=strides[1]):
            import tensorflow as tf
            from tensorflow.keras import backend as K

            def convolve_lambda_biased(i, k, b):
                import tensorflow as tf

                conv = tf.nn.conv2d(
                    i,
                    k,
                    strides=[1, stride_y, stride_x, 1],
                    dilations=[1, dilation, dilation, 1],
                    padding='VALID',
                    data_format='NHWC',
                )
                return tf.nn.bias_add(conv, b, data_format='NHWC')

            def convolve_lambda(i, k):
                import tensorflow as tf

                return tf.nn.conv2d(
                    i,
                    k,
                    strides=[1, stride_y, stride_x, 1],
                    dilations=[1, dilation, dilation, 1],
                    padding='VALID',
                    data_format='NHWC',
                )

            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights_shape)
            if has_bias:
                bias_groups = tf.split(axis=0, num_or_size_splits=groups, value=bias)
                output_groups = [
                    convolve_lambda_biased(i, k, b) for i, k, b in zip(input_groups, weight_groups, bias_groups)
                ]
            else:
                output_groups = [convolve_lambda(i, k) for i, k in zip(input_groups, weight_groups)]

            layer = tf.concat(axis=3, values=output_groups)

            return layer

        output_layer = keras.layers.Lambda(target_layer)
        outputs = output_layer(input_layer)

    else:
        # logger.debug(f'normal conv~~~~~~~~~~~~~~~~~~~~, weight shape: {node.weights[0].shape}, out channels: '
        #              f'{out_channels}, in_channels: {in_channels}, '
        #              f'Kernel_size: ({height}, {width}). '
        #              f'Groups: {n_groups}'
        #              f'Dilation rate: {dilation}')
        # logger.debug(f'Node: {node}')
        # logger.debug(f'Input shape: {input_layer.shape}, weight shape: {weights.shape}')
        output_layer = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=(height, width),  # filters
            strides=(strides[0], strides[1]),
            padding='valid',
            dilation_rate=dilation,
            use_bias=has_bias,
            weights=[weights, bias] if has_bias else [weights],
            activation=None,
        )
        # print(f'Weights ------ {node.name}: input layer: {input_layer}, '
        #       f'layer: {output_layer},')
        # print(f'Inputs:')
        # for d in inputs:
        #     print(f'SHAPE: {d.shape}, stride: {strides[0], strides[1]}')
        outputs = output_layer(input_layer)

    return outputs, output_layer


@converter('ConvTranspose')
def conv_transpose(node: OnnxNode, input_layer, *node_inputs):
    """
    Convert the add operation
    @credit to onnx2keras where I got the implementation details from:
    Link: https://github.com/gmalivenko/onnx2keras
    Args:
        node: The node that we wish to convert
    TODO: Work on this

    """
    attributes: t.Dict = node.attributes
    weights_shape = node.weights[0].shape
    filter_count = weights_shape[-2]
    padding = (
        'same'
        if attributes['pads'][0] != 0 and attributes['pads'][1] != 0 and attributes['pads'][1] == attributes['pads'][0]
        else 'valid'
    )

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
