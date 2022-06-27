import logging

from tensorflow import keras

from ..graph import OnnxNode
from .common import converter

_VALID_PAD_MODES = ('constant', 'reflect', 'edge')


@converter('Pad')
def pad(node: OnnxNode, input_layer, *inputs):
    logger = logging.getLogger(f'{__name__}::Pad')
    attr = node.attributes
    pads = attr['pads'] if 'pads' in attr else inputs[1]
    if 'mode' not in attr:
        raise ValueError(f'"mode" should be defined for node: {node}')
    mode = attr['mode']
    # we get input the following format: b'constant'
    mode = mode.decode('ascii')
    if mode not in _VALID_PAD_MODES:
        raise ValueError(f'Invalid Pad mode. Valid pad modes: {",".join(_VALID_PAD_MODES)}')

    name = f'{node.name}_Pad'
    output = None
    output_layer = None

    if mode == 'constant':

        if 'value' in attr and attr['value'] != 0.0:
            raise AssertionError('Cannot convert non-zero padding')

        # Magic ordering
        if len(pads) == 8:
            output_layer = keras.layers.ZeroPadding2D(
                padding=((pads[2], pads[6]), (pads[3], pads[7])),
                name=name,
            )
        else:
            logger.warning('Caution - no test yet')
            output_layer = keras.layers.ZeroPadding3D(
                padding=((pads[2], pads[7]), (pads[3], pads[8]), (pads[4], pads[9])),
                name=name,
            )
        output = output_layer(input_layer)
    elif mode == 'reflect':

        def target_layer(x, pads=pads):
            import tensorflow as tf

            if len(pads) == 8:
                layer = tf.pad(x, [[0, 0], [0, 0], [pads[2], pads[6]], [pads[3], pads[7]]], 'REFLECT')
            else:
                logger.warning('Caution - no test yet')
                layer = tf.pad(
                    x, [[0, 0], [0, 0], [pads[2], pads[7]], [pads[3], pads[8]], [pads[4], pads[9]]], 'REFLECT'
                )
            return layer

        output_layer = keras.layers.Lambda(target_layer, name=name)
        output = output_layer(input_layer)

    elif mode == 'edge':

        def target_layer(x, pads=pads):
            import tensorflow as tf

            # Not yet tested
            if len(pads) == 8:
                layer = tf.pad(x, [[0, 0], [0, 0], [pads[2], pads[6]], [pads[3], pads[7]]], 'SYMMETRIC')
            else:
                logger.warning('Caution - no test yet')
                layer = tf.pad(
                    x, [[0, 0], [0, 0], [pads[2], pads[7]], [pads[3], pads[8]], [pads[4], pads[9]]], 'SYMMETRIC'
                )
            return layer

        output_layer = keras.layers.Lambda(target_layer, name=name)
        output = output_layer(input_layer)

    return output, None
