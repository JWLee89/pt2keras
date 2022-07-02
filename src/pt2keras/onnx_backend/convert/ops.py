import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from ..graph import OnnxNode
from ..util import tensor_proto_to_tf_dtype
from .common import converter


@converter('Identity')
def identity(node: OnnxNode, input_layer, *inputs):
    return inputs, None


@converter('Constant')
def constant(node: OnnxNode, input_layer, *inputs):
    """
    A operation that outputs the input
    Args:
        node: The node that we wish to convert
    Returns:
    """
    return input_layer, None


@converter('Add')
def add(node: OnnxNode, input_layer, lhs, rhs):
    """
    Args:
        node: The current node inside of the onnx computational graph
        input_layer: The input layer. Since there are two inputs,
        this value will be the first input layer (LHS)
        lhs: The left hand side value. E.g.
        in 2 + 4, lhs will be 2
        rhs: The right hand side value

    Returns:
        The keras layer with the computed output node
    """
    logger = logging.getLogger('ops::Add')
    try:
        if not isinstance(lhs, np.ndarray) and not isinstance(rhs, np.ndarray):
            output_layer = keras.layers.Add()
            output = output_layer([lhs, rhs])
        else:
            raise ValueError('Operands are different.')

    except (IndexError, ValueError):
        logger.debug('Failed to use keras.layers.Add. Fallback to TF lambda.')

        def target_layer(x):
            # Import statement needs to be included when exporting models
            # to another format such as EdgeTPU
            import tensorflow as tf

            layer = tf.add(x[0], x[1])
            return layer

        output_layer = keras.layers.Lambda(target_layer)
        output = output_layer([lhs, rhs])

    return output, output_layer


@converter('Mul')
def multiply(node: OnnxNode, input_layer, lhs, rhs):
    """
    TODO: add unit test
    Args:
        node: The current computational node
        input_layer: The input layer. Since we need two input layers, this value
        will be ignored.
        lhs: The left hand side value. E.g.
        in 2 + 4, lhs will be 2
        rhs: The right hand side value
    Returns:

    """
    logger = logging.getLogger('ops::Mul')
    try:
        output_layer = keras.layers.Multiply()
        output = output_layer(lhs, rhs)
    except (IndexError, ValueError):
        logger.debug('Failed to use keras.layers.Multiply. Fallback to TF lambda.')

        # Doesn't work with constants
        # IndexError: tuple index out of range
        def target_layer(x):
            import tensorflow as tf

            layer = tf.multiply(x[0], x[1])
            return layer

        output_layer = keras.layers.Lambda(target_layer, name=f'mul_{node.name}')
        output = output_layer([lhs, rhs])
    return output, output_layer


@converter('Div')
def divide(node: OnnxNode, input_layer, lhs, rhs):
    logger = logging.getLogger('ops::Div')
    try:
        output_layer = None
        output = lhs / rhs
    except (IndexError, ValueError):
        logger.debug('Failed to use divide. Fallback to TF Lmbda')

        # Doesn't work with constants
        # IndexError: tuple index out of range
        def target_layer(x):
            import tensorflow as tf

            layer = tf.divide(x[0], x[1])
            return layer

        output_layer = keras.layers.Lambda(target_layer)
        output = output_layer([lhs, rhs])
    return output, output_layer


@converter('Cast')
def cast(node: OnnxNode, input_layer, *args):
    """
    Floor divide is considered a Cast operation in onnx,
    since we are casting from float32 to int
    """
    tf_dtype = tensor_proto_to_tf_dtype(node.attributes['to'])
    outputs = K.cast(input_layer, dtype=tf_dtype)
    return outputs, None


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
    return tf.gather(input_tensor, indices=indices, axis=mapped_axis), None


@converter('Dropout')
def dropout(node: OnnxNode, input_layer, input_tensor):
    # TODO: Dropout removed during evaluation phase
    return keras.layers.Dropout()(input_layer), keras.layers.Dropout()


@converter('Gemm')
def gemm(node, input_layer, *input_tensor):
    """
    Implementation for General Matrix Multiplication
    """
    attributes = node.attributes
    dense_layer = None
    if len(input_tensor) == 3:
        has_bias = True
        keras_weights = [input_tensor[1], input_tensor[2]]
    # Check if Bias available
    elif len(input_tensor) == 2:
        has_bias = True
        keras_weights = [input_tensor[0], input_tensor[1]]
    elif len(input_tensor) == 1:
        has_bias = False
        keras_weights = [input_tensor[0]]
    else:
        raise AttributeError('More than 3 or less than 2 inputs')

    # Linear can have additional flag to transpose weights
    if 'transB' in attributes and attributes['transB'] == 1:
        keras_weights[0] = keras_weights[0].transpose()

    # Estimate input/output neurons
    input_channels, output_channels = keras_weights[0].shape

    if isinstance(keras_weights[0], np.ndarray):
        dense_layer = tf.keras.layers.Dense(
            output_channels,
            weights=keras_weights,
            bias_initializer='zeros',
            kernel_initializer='zeros',
            use_bias=has_bias,
        )

        # The first input - always X
        try:
            output = dense_layer(input_layer)
        except ValueError:
            input_channels, output_channels = keras_weights[0].shape
            reshape = tf.keras.layers.Reshape([input_channels])
            reshaped_x = reshape(input_layer)
            output = dense_layer(reshaped_x)

    else:
        dense_layer = tf.keras.layers.Multiply()
        output = tf.keras.layers.Multiply()(input_layer, keras_weights[0])

    return output, dense_layer


@converter('MatMul')
def mat_mul(node: OnnxNode, input_layer, *inputs):
    def mat_mul_lambda(a, b):
        if not isinstance(a, np.ndarray):
            a = a.numpy()
        if not isinstance(b, np.ndarray):
            b = b.numpy()
        return np.matmul(a, b)

    # output_layer = keras.layers.Dense(inputs[1].shape[-1], use_bias=False)
    output_layer = keras.layers.Lambda(lambda t: mat_mul_lambda(t[0], t[1]))
    output = output_layer(input_layer)

    return output, output_layer
