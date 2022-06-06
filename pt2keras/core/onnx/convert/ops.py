import logging

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from .common import converter
from ..graph import OnnxNode
from ..util import tensor_proto_to_tf_dtype


@converter('Constant')
def constant(node: OnnxNode, input_layer, *inputs):
    """
    A operation that outputs the input
    Args:
        node: The node that we wish to convert
    Returns:
    """
    return input_layer


@converter('Add')
def add(node: OnnxNode, input_layer, lhs, rhs):
    logger = logging.getLogger('ops::Add')
    try:
        if not isinstance(lhs, np.ndarray) and not isinstance(rhs, np.ndarray) :
            add = keras.layers.Add()
            output = add([lhs, rhs])
        else:
            raise ValueError('Operands are different.')

    except (IndexError, ValueError):
        logger.debug('Failed to use keras.layers.Add. Fallback to TF lambda.')
        def target_layer(x):
            # Import statement needs to be included when exporting models
            # to another format such as EdgeTPU
            import tensorflow as tf
            print(x[0], x[1])
            layer = tf.add(
                x[0],
                x[1]
            )
            return layer

        lambda_layer = keras.layers.Lambda(target_layer)
        output = lambda_layer([lhs, rhs])

    return output


@converter('Mul')
def multiply(node: OnnxNode, input_layer, lhs, rhs):
    logger = logging.getLogger('ops::Mul')
    try:
        mul = keras.layers.Multiply()
        output = mul([lhs, rhs])
    except (IndexError, ValueError):
        logger.debug('Failed to use keras.layers.Multiply. Fallback to TF lambda.')
        # Doesn't work with constants
        # IndexError: tuple index out of range
        def target_layer(x):
            import tensorflow as tf
            layer = tf.multiply(
                x[0],
                x[1]
            )
            return layer

        lambda_layer = keras.layers.Lambda(target_layer)
        output = lambda_layer([lhs, rhs])
    return output


@converter('Div')
def divide(node: OnnxNode, input_layer, lhs, rhs):
    logger = logging.getLogger('ops::Div')
    try:
        output = lhs / rhs
    except (IndexError, ValueError):
        logger.debug('Failed to use divide. Fallback to TF Lmbda')

        # Doesn't work with constants
        # IndexError: tuple index out of range
        def target_layer(x):
            import tensorflow as tf
            layer = tf.divide(
                x[0],
                x[1]
            )
            return layer
        lambda_layer = keras.layers.Lambda(target_layer)
        output = lambda_layer([lhs, rhs])
    return output


@converter('Cast')
def cast(node: OnnxNode, input_layer, *args):
    """
    Floor divide is considered a Cast operation in onnx,
    since we are casting from float32 to int
    """
    tf_dtype = tensor_proto_to_tf_dtype(node.attributes['to'])
    outputs = K.cast(input_layer, dtype=tf_dtype)
    return outputs


@converter('Gather')
def gather(node: OnnxNode, input_layer, input_tensor, indices):
    # the axis to slice across
    axis = node.attributes['axis']
    # Mapping PyTorch channels to keras
    if len(input_tensor.shape) > 2:
        axis_mapper = {
            0: 0,
            1: 3,
            2: 1,
            3: 2,
        }
        mapped_axis = axis_mapper[axis]
    else:
        mapped_axis = axis
    return tf.gather(input_tensor, indices=indices, axis=mapped_axis)


@converter('Dropout')
def dropout(node: OnnxNode, input_layer, input_tensor):
    # TODO: Dropout removed during evaluation phase
    return keras.layers.Dropout()(input_layer)


@converter('Flatten')
def flatten(node: OnnxNode, input_layer, input_tensor):
    return keras.layers.Flatten()(input_layer)


@converter('Gemm')
def gemm(node: OnnxNode, input_layer, *input_tensor):
    """
    Implementation for General Matrix Multiplication
    """
    attributes = node.attributes
    # Check if Bias available
    if len(input_tensor) == 3:
        has_bias = True
        keras_weights = [input_tensor[1], input_tensor[2]]
    elif len(input_tensor) == 2:
        has_bias = False
        keras_weights = [input_tensor[1]]
    else:
        raise AttributeError('More than 3 or less than 2 inputs')

    # Linear can have additional flag to transpose weights
    if 'transB' in attributes and attributes['transB'] == 1:
        keras_weights[0] = keras_weights[0].transpose()

    # Estimate input/output neurons
    input_channels, output_channels = keras_weights[0].shape

    if isinstance(keras_weights[0], np.ndarray):
        dense = keras.layers.Dense(
            output_channels,
            weights=keras_weights,
            bias_initializer='zeros', kernel_initializer='zeros',
            use_bias=has_bias
        )

        # The first input - always X
        try:
            output = dense(input_layer)
        except ValueError:
            reshape = keras.layers.Reshape([input_channels])
            reshaped_x = reshape(input_layer)
            output = dense(reshaped_x)

    else:
        output = keras.layers.Multiply()(input_layer, keras_weights[0])

    return output
