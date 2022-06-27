import logging
import typing as t
import warnings

import numpy as np
import onnxruntime
import tensorflow as tf
import torch
import torch.nn as nn

from ..util import pt_input_to_keras

_LOGGER = logging.getLogger(__name__)
np.set_printoptions(formatter={'float': lambda x: f'{x:0.9f}'})


def test_model_output(
    source_model: t.Union[nn.Module, onnxruntime.InferenceSession],
    keras_model: tf.keras.Model,
    pt_input_shape: t.Tuple,
    strict: bool = False,
    atol=1e-4,
) -> None:
    """
    Compare and test the PyTorch and Keras model for output equality.
    An error will be asserted if the generated inputs are not close
    or are of different dimensions.
    Args:
        source_model: The source PyTorch / Onnx model
        keras_model: The target / generated Keras model
        pt_input_shape: The dimension of the PyTorch input data
        atol: The absolute tolerance parameter specified in numpy.
        See numpy documentation for more information
    """
    # Create TensorFlow input
    random_tensor_source = np.random.randn(*pt_input_shape).astype(np.float32)
    # Create Keras input tensor and convert to tensorflow tensor
    random_tensor_keras = pt_input_to_keras(random_tensor_source)
    x_keras = tf.convert_to_tensor(random_tensor_keras)

    output_keras = keras_model(x_keras)

    # Handle inputs :)
    # PyTorch / onnx inference
    if isinstance(source_model, nn.Module):
        x_pt = torch.from_numpy(random_tensor_source)
        output_source = source_model(x_pt)
    # Onnxruntime inference
    elif isinstance(source_model, onnxruntime.InferenceSession):
        output_source = source_model.run(None, {'input_0': random_tensor_source})
        # Onnx output is a list, so we flatten it if output length is 1
        if len(output_source) == 1:
            output_source = output_source[0]
    else:
        raise ValueError('Source model must be a PyTorch model on onnxruntime InferenceSession')

    # For multiple outputs
    if isinstance(output_source, (t.Tuple, t.List)):
        for source_tensor, keras_tensor in zip(output_source, output_keras):
            if isinstance(source_tensor, torch.Tensor):
                source_tensor = source_tensor.detach().cpu().numpy()
            is_approximately_equal(source_tensor, keras_tensor.numpy(), atol, strict=strict)
    # Single outputs
    else:
        if isinstance(output_source, torch.Tensor):
            output_source = output_source.detach().cpu().numpy()
        is_approximately_equal(output_source, output_keras, atol, strict=strict)


def is_approximately_equal(
    output_source: np.ndarray, output_keras: np.ndarray, atol: float = 1e-4, node=None, strict: bool = False
):
    """
    Test the outputs of the two models for equality.
    Args:
        output_source: The output of a PyTorch model.
        output_keras: The output of the converted Keras model
        atol: The absolute tolerance parameter specified in numpy.
        node: The node that we are testing. For debugging purposes
        strict: If set to true, a strict check will be applied. Otherwise, a warning will be thrown
    """
    # Convert the PyTorch / onnx model into keras output format
    # E.g. (B, C, H, W) -> (B, H, W, C)
    if len(output_source.shape) == 4:
        output_source = output_source.transpose((0, 2, 3, 1))

    # batch dimension may have been removed for PyTorch model using flatten
    if len(output_source.shape) == len(output_keras.shape) - 1:
        for pt_dim, keras_dim in zip(output_source.shape, output_keras.shape[1:]):
            assert pt_dim == keras_dim, (
                'Batch dimension may have been removed from ONNX model, but '
                f'the input dimensions still dont match. '
                f'ONNX shape: {output_source.shape}'
                f'Keras shape: {output_keras.shape}'
            )
        _LOGGER.warning(
            'Batch dimension may have possibly been removed from PyTorch model. '
            'Does your model use nn.Flatten() or torch.flatten() with start_dim=1 ?'
        )
    else:
        error_msg = (
            'ONNX and Keras model output shape should be equal. '
            f'ONNX shape: {output_source.shape}, '
            f'Keras shape: {output_keras.shape}, '
        )
        if node:
            error_msg += f'{node}'
        assert output_source.shape == output_keras.shape, error_msg

    # Average diff over all axis
    average_diff = np.mean(output_source - output_keras)

    output_is_approximately_equal = np.allclose(output_source, output_keras, atol=atol)
    assertion_error_msg = f'PyTorch output and Keras output is different. Mean difference: {average_diff}.'
    # Append useful node metadata for debugging onnx conversion operation
    if node:
        assertion_error_msg += f'\n. Node: {node}\n'
        assertion_error_msg += f'onnxruntime: {output_source}\n'
        assertion_error_msg += f'keras: {output_keras}\n'

    if strict:
        assert output_is_approximately_equal, assertion_error_msg
    elif not output_is_approximately_equal:
        warnings.warn(f'The output shows some difference with atol: {atol}. Mean diff: {average_diff}')
        warnings.warn(f'ONNX tensor: {output_source}')
        warnings.warn(f'Keras tensor: {output_keras}')
