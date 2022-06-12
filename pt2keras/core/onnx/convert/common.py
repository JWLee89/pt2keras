import logging
import typing as t
from functools import wraps

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort

from pt2keras.core.onnx.graph import Graph, OnnxNode, TestResults
from pt2keras.core.onnx.util import keras_4d_to_pt_shape, test_equality

_LOGGER = logging.getLogger('onnx:converter')


class DuplicateOperatorConverterError(Exception):
    """
    Simple exception class that is raised when the same operator
    is registered twice.
    This can create unpredictable behavior in the app and thus, an
    error is raised to minimize human error. To sidestep this error,
    create a converter with the override=True explicitly set to mark that
    the user is fully aware that they are overriding an existing converter.
    """
    pass


def _test_operation(node: OnnxNode, input_keras_layer, output_keras_layer, *inputs) -> bool:
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
    global _padding_offset_width, _padding_offset_height
    if output_keras_layer is None:
        if node.op_type == 'Constant':
            _LOGGER.debug(f'Skipping test for Constant node: {node}')
        else:
            _LOGGER.warning(f'Output keras layer not available for: {node}. '
                            f'Skipping test.')
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
    node_def = helper.make_node(node.op_type,
                                name=node.name,
                                inputs=node.input_nodes,
                                outputs=node.output_nodes,
                                **attrs)

    # Create graph input node
    input_nodes = []
    # For storing input dict for onnx runtime inference
    input_dict = {}
    start_index = 0
    if len(node.input_nodes) - 1 == len(inputs):
        input_shape = keras_4d_to_pt_shape(input_keras_layer)
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
                input_shape = keras_4d_to_pt_shape(data_to_input)
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

    if node_to_update:
        input_dict[key] = np.random.rand(*keras_4d_to_pt_shape(input_keras_layer)).astype(np.float32)
        onnx_tensor = input_dict[key]

    if len(input_keras_layer.shape) == 4:
        keras_input_data = onnx_tensor.transpose((0, 2, 3, 1))
    else:
        keras_input_data = onnx_tensor
    print(f'Keras input data: {keras_input_data.shape}, onnx input data: {onnx_tensor.shape}')
    print(f'Node name: {node_def.name}, input layer: {input_keras_layer}')
    print(f'Inputs for {node.name}: ')
    for d in inputs:
        print(f'Input: {d.shape}')

    keras_output = output_keras_layer(keras_input_data).numpy()

    # Create graph output node
    output_nodes = []
    for i, output in enumerate(node.output_nodes):
        output_shape = keras_4d_to_pt_shape(keras_output)
        value = helper.make_tensor_value_info(output, onnx.AttributeProto.FLOAT, output_shape)
        output_nodes.append(value)

    # # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],                                 # nodes
        'test-model',                               # name
        input_nodes,                                # inputs
        output_nodes,                               # outputs
    )

    # Create Model
    model_def = helper.make_model(graph_def, producer_name='test_layer')
    # Set opset version. Hardcode this Jawn for now
    model_def.opset_import[0].version = 13
    onnx.checker.check_model(model_def)

    # Prepare session for inference
    onnx_session = ort.InferenceSession(model_def.SerializeToString())
    onnx_output = onnx_session.run(None, input_dict)

    if len(onnx_output) == 1:
        onnx_output = onnx_output[0]

    print(f'Onnx output: {onnx_output.shape}, keras output: {keras_output.shape}')

    # test_equality(onnx_output, keras_output)

    return True


def converter(onnx_op: str,
              override: bool = False,
              op_testing_fn: t.Callable = None):
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
        _LOGGER.warning(f'WARNING:: Overriding existing onnx node converter: {onnx_op}. '
                        f'Please double check to ensure that this is the desired behavior.')

    if not override and onnx_op in Graph._SUPPORTED_OPERATIONS:
        raise DuplicateOperatorConverterError(f'{onnx_op} converter already exists ...')

    def inner(wrapped_fn: t.Callable) -> t.Callable:
        """
        The inner workings on the decorator
        Args:
            wrapped_fn: The converter function we are wrapping

        Returns:

        """
        @wraps(wrapped_fn)
        def created_converter(onnx_node: OnnxNode, input_layer, computational_graph, node_dict: t.Dict,
                              test_results: TestResults, *args, **kwargs) -> t.Any:
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
            test_layer: t.Callable = _test_operation if op_testing_fn is None else op_testing_fn
            is_tested = test_layer(onnx_node, input_layer, keras_layer, *args, **kwargs)

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
