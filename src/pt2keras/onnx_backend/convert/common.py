import logging
import time
import typing as t
from functools import wraps

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

from ..graph import Graph, OnnxNode, TestResults
from ..testing.utils import is_approximately_equal
from ..util import keras_input_to_pt

_LOGGER = logging.getLogger('onnx:converter')


class DuplicateOperatorError(Exception):
    """
    Simple exception class that is raised when the same operator
    is registered twice.
    This can create unpredictable behavior in the app and thus, an
    error is raised to minimize human error. To sidestep this error,
    create a converter with the override=True explicitly set to mark that
    the user is fully aware that they are overriding an existing converter.
    """

    pass


def _test_double_input_operation(node: OnnxNode, opset_version, input_keras_layer, output_keras_layer, *inputs) -> bool:

    if len(inputs) != 2:
        raise ValueError(f'Expected two inputs, received: {len(inputs)} inputs')

    # create the computational node graph
    node_def = helper.make_node(
        node.op_type, name=node.name, inputs=node.input_nodes, outputs=node.output_nodes, **node.attributes
    )

    # Create graph input node
    input_nodes = []
    # For storing input dict for onnx runtime inference
    input_dict = {}
    for i in range(len(node.input_nodes)):
        input_node_name = node.input_nodes[i]
        onnx_input_data = inputs[i]
        input_shape = onnx_input_data.shape
        if not isinstance(onnx_input_data, np.ndarray):
            if len(input_shape) == 4:
                data_to_input = onnx_input_data
                input_shape = keras_input_to_pt(data_to_input).shape
            else:
                data_to_input = onnx_input_data
                input_shape = onnx_input_data.shape
        else:
            data_to_input = onnx_input_data
            input_shape = onnx_input_data.shape
        # Do not add numpy data again if it is not needed.
        input_dict[input_node_name] = data_to_input
        value = helper.make_tensor_value_info(input_node_name, onnx.AttributeProto.FLOAT, input_shape)
        input_nodes.append(value)

    onnx_input_dict = {}
    keras_input_list = []
    for key, value in input_dict.items():
        onnx_input_dict[key] = np.random.rand(*keras_input_to_pt(value).shape).astype(np.float32)
        keras_input = onnx_input_dict[key]
        if len(onnx_input_dict[key].shape) >= 4:
            keras_input = keras_input.transpose((0, 2, 3, 1))

        keras_input_list.append(keras_input)

    keras_start_time = time.monotonic()
    keras_output = output_keras_layer(keras_input_list)
    keras_runtime_ms = (time.monotonic() - keras_start_time) * 1000
    keras_output = keras_output.numpy()

    # Create graph output node
    output_nodes = []
    for i, output in enumerate(node.output_nodes):
        output_shape = keras_input_to_pt(keras_output).shape
        value = helper.make_tensor_value_info(output, onnx.AttributeProto.FLOAT, output_shape)
        output_nodes.append(value)

    # # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],  # nodes
        'test-model',  # name
        input_nodes,  # inputs
        output_nodes,  # outputs
    )

    # Create Model
    model_def = helper.make_model(graph_def, producer_name='test_layer')
    # Set opset version. Hardcode this Jawn for now
    model_def.opset_import[0].version = opset_version
    onnx.checker.check_model(model_def)

    # Prepare session for inference
    onnx_session = ort.InferenceSession(model_def.SerializeToString())
    onnx_start_time = time.monotonic()
    onnx_output = onnx_session.run(None, onnx_input_dict)
    onnx_runtime_ms = (time.monotonic() - onnx_start_time) * 1000

    _LOGGER.debug(
        f'Node: {node.name}, op: {node.op_type}. \n'
        f'onnxruntime speed: {onnx_runtime_ms} ms\n'
        f'keras speed: {keras_runtime_ms} ms'
    )

    if len(onnx_output) == 1:
        onnx_output = onnx_output[0]

    is_approximately_equal(onnx_output, keras_output, node=node)
    return True


def _test_operation(node: OnnxNode, opset_version, input_keras_layer, output_keras_layer, *inputs) -> bool:
    """
    The default testing function. After each layer / operation is made,
    the test will be run to ensure that the output from the onnx model
    is identical to the converted Keras model.
    Args:
        node: A class representation of the onnx node.
        input_keras_layer: The input keras layer
        output_keras_layer: The output of this keras layer must be tested.

    Returns:
        True if tested, otherwise return False.
    """
    logger = logging.getLogger(f'{__name__}._test_operation')
    if output_keras_layer is None:
        if node.op_type == 'Constant':
            logger.debug(f'Skipping test for Constant node: {node}')
        else:
            logger.warning(f'Output keras layer not available. Skipping test for node: \n {node}. ')
        return False

    # Create attributes for making node for onnxruntime inference
    attrs = {}
    for key, attribute_val in node.attributes.items():
        if key != 'name' and key != 'inputs' and key != 'outputs':
            # we need to add extra padding ton compensate for the zero-padding added to the model
            # by keras which is then added again by Onnx
            # So therefore, we ignore it
            if key == 'pads':
                continue
            attrs[key] = attribute_val

    # create the computational node graph
    node_def = helper.make_node(
        node.op_type, name=node.name, inputs=node.input_nodes, outputs=node.output_nodes, **attrs
    )

    # Create graph input node
    input_nodes = []
    # For storing input dict for onnx runtime inference
    input_dict = {}
    start_index = 0
    if len(node.input_nodes) - 1 == len(inputs):
        input_shape = keras_input_to_pt(input_keras_layer).shape
        node_name = node.input_nodes[0]
        value = helper.make_tensor_value_info(node_name, onnx.AttributeProto.FLOAT, input_shape)
        input_dict[node_name] = np.random.rand(*input_shape).astype(np.float32)
        input_nodes.append(value)
        start_index += 1

    for i in range(start_index, len(node.input_nodes)):
        input_node_name = node.input_nodes[i]
        onnx_input_data = inputs[i - start_index]
        input_shape = onnx_input_data.shape
        if not isinstance(onnx_input_data, np.ndarray):
            if len(input_shape) == 4:
                data_to_input = onnx_input_data
                input_shape = keras_input_to_pt(data_to_input).shape
            else:
                data_to_input = onnx_input_data
                input_shape = onnx_input_data.shape
        else:
            data_to_input = onnx_input_data
            input_shape = onnx_input_data.shape
        # Do not add numpy data again if it is not needed.
        input_dict[input_node_name] = data_to_input
        value = helper.make_tensor_value_info(input_node_name, onnx.AttributeProto.FLOAT, input_shape)
        input_nodes.append(value)

    node_to_update = None
    onnx_tensor = None
    for key, value in input_dict.items():
        if not isinstance(value, np.ndarray):
            node_to_update = key
        else:
            onnx_tensor = value
        break

    # TODO: Make sure that we can receive multiple inputs
    if node_to_update:
        input_dict[key] = np.random.randn(*keras_input_to_pt(input_keras_layer).shape).astype(np.float32)
        onnx_tensor = input_dict[key]

    if len(input_keras_layer.shape) == 4:
        keras_input_data = onnx_tensor.transpose((0, 2, 3, 1))
    else:
        keras_input_data = onnx_tensor

    keras_start_time = time.monotonic()
    keras_output = output_keras_layer(keras_input_data)
    keras_runtime_ms = (time.monotonic() - keras_start_time) * 1000
    keras_output = keras_output.numpy()

    # Create graph output node
    output_nodes = []
    for i, output in enumerate(node.output_nodes):
        output_shape = keras_input_to_pt(keras_output).shape
        value = helper.make_tensor_value_info(output, onnx.AttributeProto.FLOAT, output_shape)
        output_nodes.append(value)

    # # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],  # nodes
        'test-model',  # name
        input_nodes,  # inputs
        output_nodes,  # outputs
    )

    try:
        # Create Model
        model_def = helper.make_model(graph_def, producer_name='test_layer')
        model_def.opset_import[0].version = opset_version
        onnx.checker.check_model(model_def)

        # Prepare session for inference
        onnx_session = ort.InferenceSession(model_def.SerializeToString())
        onnx_start_time = time.monotonic()
        onnx_output = onnx_session.run(None, input_dict)
        onnx_runtime_ms = (time.monotonic() - onnx_start_time) * 1000

        logger.debug(
            f'Node: {node.name}, op: {node.op_type}. \n'
            f'onnxruntime speed: {onnx_runtime_ms} ms\n'
            f'keras speed: {keras_runtime_ms} ms'
        )

        if len(onnx_output) == 1:
            onnx_output = onnx_output[0]

        is_approximately_equal(onnx_output, keras_output, node=node)
        return True
    except Exception as ex:
        logger.warning(
            f'Error while creating automated test for Layer: {output_keras_layer}.\n '
            f'node: {node}. \n'
            f'Exception: {ex}'
        )
        return False


def converter(onnx_op: str, override: bool = False, op_testing_fn: t.Callable = None):
    """
    Args:
        onnx_op: An onnx operation that we wish to add a converter for

        Optional args:

        override: If true, will override existing converter if it exists.
        op_testing_fn: If included, it will override the default testing function.
        This is useful if you wish to construct custom tests that best suit specific
        use cases.
    """
    # TODO add check to see whether operator is valid

    if override:
        _LOGGER.warning(
            f'WARNING:: Overriding existing onnx node converter: {onnx_op}. '
            f'Please double check to ensure that this is the desired behavior.'
        )
    if not override and onnx_op in Graph._SUPPORTED_OPERATIONS:
        raise DuplicateOperatorError(f'Converter for "{onnx_op}" already exists ...')

    def inner(wrapped_fn: t.Callable) -> t.Callable:
        """
        The inner workings on the decorator
        Args:
            wrapped_fn: The converter function we are wrapping
        """

        @wraps(wrapped_fn)
        def created_converter(
            onnx_node: OnnxNode,
            input_layer,
            computational_graph: t.Dict,
            opset_version: int,
            test_results: TestResults,
            *args,
            **kwargs,
        ) -> t.Any:
            """
            Given a pytorch operation or layer, directly port it to keras
            Returns:
                The converted keras layer with all states copied / patched onto the keras layer
            """

            # Should add all available arguments and so on
            _LOGGER.debug(f'Converting: {onnx_op}. Node name: {onnx_node.name}')
            output = wrapped_fn(onnx_node, input_layer, *args, **kwargs)

            # Layer not outputted. Fill as None
            # If keras layer is not made, we will not be doing any test
            if not isinstance(output, t.Tuple):
                keras_tensor = output
                keras_layer = None
            elif len(output) == 3:
                keras_tensor, input_layer, keras_layer = output
            elif len(output) == 2:
                keras_tensor, keras_layer = output
            else:
                raise ValueError(f'Invalid output format for converter: "{onnx_node.op_type}"')

            # build computational graph during conversion and forward pass
            for output_node_name in onnx_node.output_nodes:
                if output_node_name not in computational_graph:
                    computational_graph[output_node_name] = keras_tensor

            # Post processing
            # -------------------

            # 1. Perform tests to see whether the two layers (pytorch and keras)
            # are outputting the same value and shape
            if onnx_node.op_type in ['Mul', 'Add', 'Sub', 'Div']:
                test_layer: t.Callable = _test_double_input_operation if op_testing_fn is None else op_testing_fn
            else:
                test_layer: t.Callable = _test_operation if op_testing_fn is None else op_testing_fn

            is_tested = test_layer(onnx_node, opset_version, input_layer, keras_layer, *args, **kwargs)

            # Add test data
            if is_tested:
                test_results.tested_nodes.append(onnx_node.name)
                test_results.tested_operations.add(onnx_node.op_type)
            else:
                test_results.untested_nodes.append(onnx_node.name)
                test_results.untested_operations.add(onnx_node.op_type)

            return keras_tensor

        if not override:
            _LOGGER.info(f'Registering onnx node converter: {onnx_op}')
        Graph._SUPPORTED_OPERATIONS[onnx_op] = created_converter
        return created_converter

    return inner
