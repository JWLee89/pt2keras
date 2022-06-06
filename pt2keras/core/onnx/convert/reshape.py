import logging

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from .common import converter
from ..graph import OnnxNode


@converter('Concat')
def concat(node: OnnxNode, input_layer, *inputs):
    """
    A operation that outputs the input
    Args:
        node: The node that we wish to convert
    Returns:
    """
    logger = logging.getLogger('onnx::Concat')
    return None, None


@converter('Reshape')
def constant(node: OnnxNode, _, *inputs):
    """
    A operation that reshapes the input array or layer.
    @credit to onnx2keras for the implementation
    link: https://github.com/gmalivenko/onnx2keras/blob/master/onnx2keras/reshape_layers.py
    Args:
        node: The node that we wish to convert
    Returns:
    """
    logger = logging.getLogger('onnx::Reshape')
    input_layer, shape_arr = inputs
    output_layer = None
    attributes = node.attributes
    if isinstance(shape_arr, np.ndarray):

        if isinstance(input_layer, np.ndarray):
            logger.debug('input layer is numpy array. Doing np.reshape')
            output = np.reshape(input_layer, np.int32(shape_arr))

        elif 'change_ordering' in attributes:

            # Fix critical issue with NHWC
            if shape_arr[0] is None and shape_arr[1] == -1:
                logger.warning('!!! IMPORTANT INFORMATION !!!')
                logger.warning('The target shape if [None, -1] that means flatten.')
                logger.warning('But the target ordering is NHWC, so we cant simply perform flatten')
                logger.warning('The layer will be converted as lambda with tf.transpose')
                logger.warning('---')

                def target_layer(x):
                    import tensorflow as tf
                    x = tf.transpose(x, [0, 3, 1, 2])
                    return x

                output_layer = keras.layers.Lambda(target_layer)
                output = output_layer(input_layer)
            else:
                output = input_layer

            output_layer = keras.layers.Reshape(np.int32(shape_arr[1:]))
            output = output_layer(output)

        else:
            logger.debug('The first argument is Keras/tf layer. Apply keras.Reshape.')
            logger.debug('Target shape :')
            logger.debug(np.int32(shape_arr[1:]))

            if len(np.int32(shape_arr[1:])) == 1 and np.int32(shape_arr[1:])[0] == -1:
                logger.debug('The first argument is Keras/tf layer. Apply keras.Flatten.')
                output_layer = keras.layers.Flatten()
                output = output_layer(input_layer)
            else:
                output_layer = keras.layers.Reshape(np.int32(shape_arr[1:]))
                output = output_layer(input_layer)
    else:
        raise AttributeError('Cannot reshape array with dynamic size')
    return output, output_layer
