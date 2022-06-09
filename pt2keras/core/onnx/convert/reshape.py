import logging

import numpy as np

from tensorflow import keras

from .common import converter
from ..graph import OnnxNode


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
    print(f'RESHAPE ------------------------ {attributes}, inputs: {inputs}')
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
                    # input dimensions for PyTorch are BCHW, whereas TF / Keras default is BHWC
                    # Change from TF / Keras Shape (BHWC) -> PyTorch (BCHW)
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
            reshape_target = np.int32(shape_arr[1:])
            logger.debug(f'Target shape : {reshape_target.shape}')
            if not reshape_target:
                reshape_target = np.int32(shape_arr)
                logger.warning(f'Removing batch dimensions ... new shape: {reshape_target} ')

            if len(reshape_target) == 1 and reshape_target[0] == -1:
                logger.debug('The first argument is Keras/tf layer. Apply keras.Flatten.')

                # IMPORTANT
                # We need to make the output channels first for the output to equal
                # Otherwise, we will be performing the wrong flattening operation T_T
                output_layer = keras.layers.Flatten()
                output = output_layer(input_layer)
            else:
                output_layer = keras.layers.Reshape(reshape_target)
                print(f'Input layer: {input_layer}, shape arr: {reshape_target}')
                output = output_layer(input_layer)
    else:
        raise AttributeError('Cannot reshape array with dynamic size')
    return output, output_layer


@converter('Flatten')
def flatten(node: OnnxNode, input_layer, input_tensor):
    print(f'input tensor: {input_tensor}')
    return keras.layers.Flatten()(input_layer), keras.layers.Flatten()

# @converter('Flatten')
# def flatten(node: OnnxNode, input_layer, *input_tensor):
#     logger = logging.getLogger('onnx::Flatten')
#
#     if len(input_tensor) != 1:
#         raise AttributeError('Number of inputs is not equal 1 for flatten layer')
#
#     logger.debug(f'Convert Flatten ... {node}')
#     input_tensor = input_tensor[0]
#
#     # Need to transpose, otherwise we get all sorts of funky errors ...
#     def target_layer(x):
#         import tensorflow as tf
#         x = tf.transpose(x, [0, 3, 1, 2])
#         return x
#
#     lambda_layer = keras.layers.Lambda(target_layer)
#     new_input = lambda_layer(input_tensor)
#
#     output_layer = keras.layers.Reshape([-1])
#     output = output_layer(new_input)
#
#     return output, output_layer


@converter('Shape')
def shape(node: OnnxNode, _, *inputs):
    logger = logging.getLogger('onnx::Shape')
    input_layer = inputs[0]

    logger.debug('Actual shape:')
    logger.debug(np.array(input_layer.shape))

    shapes = []
    for i in input_layer.shape:
        if i is not None:
            shapes.append(i)
        else:
            shapes.append(None)

    output = np.array(shapes)
    return output


@converter('Concat')
def concat(node: OnnxNode, _, *inputs):
    logger = logging.getLogger('onnx::Concat')

    attributes = node.attributes
    axis = attributes['axis']

    layer_input = inputs
    output_layer = None
    if all([isinstance(layer, np.ndarray) for layer in inputs]):
        logger.debug(f'Concat numpy arrays.')
        output = np.concatenate(layer_input, axis=axis)
    else:
        logger.debug('Concat Keras layers.')
        if len(layer_input) > 1:
            try:
                output_layer = keras.layers.concatenate(inputs=layer_input, axis=axis)
            except:
                logger.warning('!!! IMPORTANT INFORMATION !!!')
                logger.warning('Something goes wrong with concat layers. Will use TF fallback.')
                logger.warning('---')

                def target_layer(x, axis=axis):
                    import tensorflow as tf
                    x = tf.concat(x, axis=axis)
                    return x

                output_layer = keras.layers.Lambda(target_layer, name="%s_CHW" % node.name)
                output = output_layer(layer_input)
        else:
            output = layer_input[0]

    return output, output_layer


@converter('Slice')
def slice_inputs(node: OnnxNode, _, *inputs):
    """
    A operation that reshapes the input array or layer.
    @credit to onnx2keras for the implementation
    link: https://github.com/gmalivenko/onnx2keras/blob/master/onnx2keras/reshape_layers.py
    Args:
        node: The node that we wish to convert
    Returns:
    """
    logger = logging.getLogger('onnx::Slice')
    if len(inputs) == 5:
        input_layer, starts, ends, axes, steps = inputs
    else:
        input_layer, starts, ends, axes = inputs
        steps = None

    output_layer = None
    if isinstance(input_layer, np.ndarray):
        logger.debug('Slice numpy constants')
        starts = starts[0]
        ends = ends[0]
        axes = axes[0]

        if axes == 0:
            output = input_layer[starts:ends]
        elif axes == 1:
            output = input_layer[:, starts:ends]
        elif axes == 2:
            output = input_layer[:, :, starts:ends]
        elif axes == 3:
            output = input_layer[:, :, :, starts:ends]
        else:
            raise AttributeError(f'Slice not implemented for dimensions: {input_layer.shape}')
    else:
        logger.debug('Convert inputs to Keras/TF layers if needed.')
        logger.debug(f'Start: {starts}, ends: {ends}, axes: {axes}, steps: {steps}')

        if len(axes) != 1:
            logger.debug(f'Axes is list or numpy array: {axes}')

            def target_layer(x, axes=np.array(axes), starts=starts, ends=ends):
                import tensorflow as tf
                rank = max(axes)
                s = [0 for _ in range(rank + 1)]
                e = [0 for _ in range(rank + 1)]
                mask = 0xff
                for _s, _e, axis in zip(starts, ends, axes):
                    s[axis] = _s
                    e[axis] = _e
                    mask = mask ^ (0x1 << axis)
                return tf.strided_slice(x, s, e, begin_mask=mask, end_mask=mask)

            output_layer = keras.layers.Lambda(target_layer)
            output = output_layer(input_layer)
        else:
            logger.debug(f'Axes is a number: {axes}')

            def target_layer(x, axis=axes[0], starts=starts[0], ends=ends[0]):
                import tensorflow as tf
                rank = axis
                s = [0 for _ in range(rank + 1)]
                e = [0 for _ in range(rank + 1)]
                mask = 0xff
                s[axis] = starts
                e[axis] = ends
                mask = mask ^ (0x1 << axis)
                return tf.strided_slice(x, s, e, begin_mask=mask, end_mask=mask)

            output_layer = keras.layers.Lambda(target_layer)
            output = output_layer(input_layer)
    logger.debug(f'Output handled: {output}, Layer: {output_layer}')
    return output, output_layer