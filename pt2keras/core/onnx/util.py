import typing as t

import numpy as np
import onnx.checker
import onnx.helper
import tensorflow as tf
import torch


def test_model_output(pytorch_model: torch.nn.Module,
                      keras_model: tf.keras.Model,
                      pt_input_shape: t.Tuple,
                      keras_input_shape: t.Tuple,
                      atol=1e-4):

    x_keras = tf.ones(keras_input_shape)
    x_pt = torch.ones(pt_input_shape)

    output_pt = pytorch_model(x_pt)
    output_keras = keras_model(x_keras)

    if len(output_pt.shape) == 4:
        output_pt = output_pt.permute(0, 2, 3, 1)

    output_pt = output_pt.cpu().detach().numpy()
    output_keras = output_keras.numpy()

    assert output_pt.shape == output_keras.shape, 'PyTorch and Keras model output shape should be equal. ' \
                                                  f'Got pt: {output_pt.shape}, Keras: {output_keras.shape}'

    # Average diff over all axis
    average_diff = np.mean(np.mean([output_pt, output_keras]))

    output_is_approximately_equal = np.allclose(output_pt, output_keras, atol=atol)
    assert output_is_approximately_equal, f'PyTorch output and Keras output is different. ' \
                                          f'Mean difference: {average_diff}'


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


def get_graph_input_shape(graph: onnx.GraphProto,
                          transpose_matrix: t.Union[t.List, t.Tuple]) -> t.List[t.Dict]:
    """
    Args:
        graph: The graph we want t
        transpose_matrix:

    Returns:
        A list containing information on the graph input node
    """
    shape_info = []
    for node in graph.input:
        data = _add_node_property(node, transpose_matrix)
        shape_info.append(data)
    return shape_info


def get_graph_output_shape(graph: onnx.GraphProto,
                           transpose_matrix: t.Union[t.List, t.Tuple] = None) -> t.List[t.Dict]:
    """
    Args:
        graph: The graph we want t
        transpose_matrix:

    Returns:
        A list containing information on the graph output node
    """
    shape_info = []
    for node in graph.input:
        data = _add_node_property(node, transpose_matrix)
        shape_info.append(data)
    return shape_info


def _add_node_property(node, transpose_matrix: t.Union[t.List, t.Tuple] = None):
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


def generate_node_key(node: onnx.NodeProto):
    final_key = ''
    input_str = ''
    for input_name in node.input:
        input_str += f'{input_name}_'

    if input_str:
        final_key += 'INPUTINFO-'
        final_key += input_str

    output_str = ''
    for output_name in node.output:
        output_str += f'{output_name}_'

    # remove trailing '_'
    if output_str:
        final_key += 'OUTPUTINFO-'
        final_key += output_str

    if final_key.endswith('_'):
        final_key = final_key[:-1]

    return final_key

