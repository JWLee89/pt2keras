import logging

import numpy as np
from tensorflow import keras

from ..graph import OnnxNode
from ..util import keras_input_to_pt, to_tf
from .common import converter


@converter('Reshape', override=True)
def reshape(node, input_tensor, *inputs):
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

    # In case shape_arr is a tuple, we need to convert to np array.
    if isinstance(shape_arr, tuple):
        shape_arr = np.array(shape_arr, dtype=np.int32).reshape(-1)

    # Otherwise, if the shape_arr is tf.Tensor, we attempt to convert it to numpy
    else:
        try:
            shape_arr = shape_arr.numpy()
        except Exception:
            pass

    output_layer = None
    if isinstance(shape_arr, np.ndarray):

        if isinstance(input_layer, np.ndarray):
            logger.debug('input layer is numpy array. Doing np.reshape')
            output = np.reshape(input_layer, np.int32(shape_arr))
        else:
            logger.debug('The first argument is Keras/tf layer. Apply keras.Reshape.')
            reshape_target = np.int32(shape_arr[1:])
            logger.debug(f'Target shape : {reshape_target.shape}')
            if not reshape_target:
                reshape_target = np.int32(shape_arr)
                logger.warning(f'Removing batch dimensions ... new shape: {reshape_target} ')

            tensor_chw = keras_input_to_pt(input_layer)

            # Removing entire feature dimension. Output shape is : (feature_count,)
            if len(reshape_target) == 1 and reshape_target[0] == -1:
                output_layer = keras.layers.Flatten(name=f'{node.name}_flatten')
                output = output_layer(tensor_chw)
                return output, None

            else:
                output_layer = keras.layers.Reshape(reshape_target)
                output = output_layer(tensor_chw)
    else:
        raise AttributeError('Cannot reshape array with dynamic size')
    return output, output_layer


@converter('Flatten')
def flatten(node: OnnxNode, input_layer, *input_tensor):
    if len(input_tensor) != 1:
        raise AttributeError('Number of inputs is not equal to 1 for Flatten()')

    input_tensor = input_tensor[0]
    transpose_custom = keras.layers.Permute((3, 1, 2))
    tensor_chw = transpose_custom(input_tensor)
    output_layer = keras.layers.Flatten(name=f'{node.name}_flatten')
    output = output_layer(tensor_chw)
    return output, output_layer


@converter('Shape')
def shape(node: OnnxNode, _, *inputs):
    logger = logging.getLogger('onnx::Shape')
    input_layer = inputs[0]

    logger.debug('Actual shape:')
    logger.debug(np.array(input_layer.shape))

    shapes = []
    for i in input_layer.shape:
        # Note that the value can be "None" if it is dynamic
        shapes.append(i)

    output = np.array(shapes)
    return output


@converter('Concat')
def concat(node: OnnxNode, _, *inputs):
    logger = logging.getLogger('onnx::Concat')

    attributes = node.attributes
    axis = attributes['axis']

    # PyTorch channel. Append to end
    if axis == 1:
        axis = -1

    layer_input = inputs
    output_layer = None
    if all([isinstance(layer, np.ndarray) for layer in inputs]):
        logger.debug('Concat numpy arrays.')
        output = np.concatenate(layer_input, axis=axis)
    else:
        logger.debug('Concat Keras layers.')
        if len(layer_input) > 1:
            try:
                output_layer = keras.layers.concatenate(inputs=layer_input, axis=axis)
                output = output_layer(layer_input)
            except Exception:
                logger.warning('!!! IMPORTANT INFORMATION !!!')
                logger.warning('Something goes wrong with concat layers. Will use TF fallback.')
                logger.warning('---')

                def target_layer(x, axis=axis):
                    import tensorflow as tf

                    return tf.concat(x, axis=axis)

                output_layer = keras.layers.Lambda(target_layer, name=f'{node.name}_CHW')
                output = output_layer(layer_input)
        else:
            output = layer_input[0]

    return output, None


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
                mask = 0xFF
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
                mask = 0xFF
                s[axis] = starts
                e[axis] = ends
                mask = mask ^ (0x1 << axis)
                return tf.strided_slice(x, s, e, begin_mask=mask, end_mask=mask)

            output_layer = keras.layers.Lambda(target_layer)
            output = output_layer(input_layer)
    logger.debug(f'Output handled: {output}, Layer: {output_layer}')
    return output, output_layer


@converter('Unsqueeze')
def unsqueeze(node: OnnxNode, _, *inputs):
    input_data = inputs[0]

    def target_layer(x):
        from tensorflow import keras

        return keras.backend.expand_dims(x, 0)

    output_layer = keras.layers.Lambda(target_layer)
    output = output_layer(input_data)

    return output, output_layer


@converter('Squeeze')
def squeeze(node: OnnxNode, _, *inputs):
    input_0 = to_tf(inputs[0])
    attributes = node.attributes

    def target_layer(x, axis=attributes['axes'][0]):
        from tensorflow import keras

        return keras.backend.squeeze(x, axis)

    output_layer = keras.layers.Lambda(target_layer)
    output = output_layer(input_0)
    return output, output_layer


@converter('Transpose')
def transpose(node: OnnxNode, input_layer, *inputs):
    attributes = node.attributes
    output_layer = None
    if attributes['perm'][0] != 0:
        print('Cannot permute batch dimension. Result may be wrong.')
        if isinstance(input_layer, np.ndarray):
            print('Transposing numpy array.')
            output = np.transpose(input_layer, axes=attributes['perm'])
        else:
            raise NotImplementedError('Cannot modify this type of data')
    else:
        output_layer = keras.layers.Permute(attributes['perm'][1:])
        output = output_layer(input_layer)

    return output, output_layer


@converter('Clip')
def clip(node: OnnxNode, input_layer, *inputs):
    # Second and third attributes are min, max values
    min_val = inputs[1]
    max_val = inputs[2]

    def clip_layer(x):
        from tensorflow import keras

        return keras.backend.clip(x, min_val, max_val)

    output_layer = keras.layers.Lambda(clip_layer)
    output = output_layer(input_layer)
    return output, output_layer
