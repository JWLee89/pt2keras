import logging
import typing as t

import numpy as np
import onnx
import tensorflow as tf
import torch
from tensorflow import keras

_LOGGER = logging.getLogger(__name__)


def pt_input_to_keras(pt_shaped_input_data) -> t.Tuple:
    """
    Given an np.ndarray or keras tensor, convert the PyTorch input shape to Keras input shape
    Args:
        pt_shaped_input_data: An input tensor or np array shaped in the form of a PyTorch input

    Returns:
        A tuple representing the output dimensions in Keras format
    """
    if not isinstance(pt_shaped_input_data, np.ndarray) and not keras.backend.is_keras_tensor(pt_shaped_input_data):
        raise ValueError('Not a np.ndarray or KerasTensor')

    # Transpose function varies based on data type. can
    transpose_function = np.transpose if isinstance(pt_shaped_input_data, np.ndarray) else tf.transpose
    input_dims = len(pt_shaped_input_data.shape)
    if input_dims >= 4:
        # (B, C, H, W) -> (B, H, W, C)
        # (B, C, X, Y, Z) -> (B, X, Y, Z, C)
        # and so on
        transpose_vector = (0,) + tuple(i for i in range(2, input_dims)) + (1,)
        output_data = transpose_function(pt_shaped_input_data, transpose_vector)
    else:
        output_data = pt_shaped_input_data
    return output_data


def keras_input_to_pt(input_data: np.ndarray) -> t.Any:
    """
    Given an input data, if it is a 4d input,
    such as an image or conv intermediate feature,
    we assume that input data is Keras format
    (B, H, W, C) and we will convert to PyTorch:
    (B, C, H, W).
    Args:
        input_data: The input data we will be transforming.
    Returns:
        The converted data
    """
    if not hasattr(input_data, 'shape'):
        raise ValueError('Not an np.ndarray or KerasTensor')

    # Transpose function varies based on data type. can
    transpose_function = None
    if isinstance(input_data, np.ndarray):
        transpose_function = np.transpose
    elif isinstance(input_data, torch.Tensor):
        transpose_function = torch.permute
    elif tf.is_tensor(input_data):
        transpose_function = tf.transpose
    else:
        raise ValueError('Must be PyTorch / TF tensor or numpy array')
    input_dims = len(input_data.shape)
    if input_dims >= 4:
        # (B, C, H, W) -> (B, H, W, C)
        transpose_vector = (0, input_dims - 1) + tuple(i for i in range(1, input_dims - 1))
        return transpose_function(input_data, transpose_vector)
    return input_data


def get_tensor_data(initializer: onnx.TensorProto) -> None:
    """
    Given an onnx tensor, return the data / weights stored
    """
    if initializer.data_type == onnx.TensorProto.DataType.FLOAT:
        return initializer.float_data
    elif initializer.data_type == onnx.TensorProto.DataType.INT32:
        return initializer.int32_data
    elif initializer.data_type == onnx.TensorProto.DataType.INT64:
        return initializer.int64_data
    elif initializer.data_type == onnx.TensorProto.DataType.DOUBLE:
        return initializer.double_data
    elif initializer.data_type == onnx.TensorProto.DataType.UINT64:
        return initializer.uint64_data
    else:
        raise NotImplementedError


def tensor_proto_to_tf_dtype(tensor_data_type: int):
    """
    Given an tensor proto data type value (int), convert it to appropriate
    tensorflow data type
    """
    if tensor_data_type == onnx.TensorProto.DataType.FLOAT:
        return tf.float32
    elif tensor_data_type == onnx.TensorProto.DataType.INT32:
        return tf.int32
    elif tensor_data_type == onnx.TensorProto.DataType.INT64:
        return tf.int64
    elif tensor_data_type == onnx.TensorProto.DataType.DOUBLE:
        return tf.float64
    elif tensor_data_type == onnx.TensorProto.DataType.UINT64:
        return tf.uint64
    else:
        raise NotImplementedError


class NodeProperties:
    name = 'name'
    shape = 'shape'


def get_graph_shape_info(graph: onnx.GraphProto, transpose_list: t.Union[t.List, t.Tuple]) -> t.List[t.Dict]:
    """
    Args:
        graph: The graph we want to evaluate
        transpose_list: The transpose mapping list (see np.transpose for more information)

    Returns:
        A list containing information on the graph input node
    """
    shape_info = []
    for node in graph.input:
        data = get_node_property(node, transpose_list)
        shape_info.append(data)
    return shape_info


def to_tf(obj, fake_input_layer=None, name=None):
    """
    Convert to Keras Constant if needed
    @Credit onnx2keras for this function
    https://github.com/gmalivenko/onnx2keras/blob/45c81f221bb4228751abb061cb24d473bb74a8e8/onnx2keras/utils.py#L26

    Args:
        obj: numpy / tf type
        fake_input_layer: fake input layer to add constant
    Returns:
        Tf constant
    """
    if isinstance(obj, np.ndarray):
        # Downcast
        if obj.dtype == np.int64:
            obj = np.int32(obj)

        def target_layer(_, inp=obj, dtype=obj.dtype.name):
            import numpy as np
            import tensorflow as tf

            if not isinstance(inp, (np.ndarray, np.generic)):
                inp = np.array(inp, dtype=dtype)
            return tf.constant(inp, dtype=inp.dtype)

        lambda_layer = keras.layers.Lambda(target_layer, name=name)
        output = lambda_layer(fake_input_layer)

        return output
    return obj


def get_graph_output_info(graph: onnx.GraphProto, transpose_matrix: t.Union[t.List, t.Tuple] = None) -> t.List[t.Dict]:
    """
    Args:
        graph: The graph we want to analyze
        transpose_matrix: The tranpose matrix that we want to apply on the shape of the graph.

    Returns:
        A list containing information on the graph output node
    """
    shape_info = []
    for node in graph.output:
        data = get_node_property(node, transpose_matrix)
        shape_info.append(data)
    return shape_info


def get_node_property(node: onnx.NodeProto, transpose_matrix: t.Union[t.List, t.Tuple] = None) -> t.Dict:
    """
    Given a node and the transpose matrix, output a dictionary containing the node properties.
    Args:
        node: The node that we want to evaluate and retrieve the node property
        transpose_matrix: The transformation that we want to apply to the nodes input shape
    Returns:
        A dictionary containing the node properties.
    """
    data = {NodeProperties.name: node.name}
    input_shape = []
    # Grab dimension information
    dimensions = node.type.tensor_type.shape.dim
    if transpose_matrix:
        for index in transpose_matrix:
            input_shape.append(dimensions[index].dim_value)
    else:
        for dim in dimensions:
            input_shape.append(dim.dim_value)
    # Convert to appropriate format
    data[NodeProperties.shape] = input_shape
    return data
