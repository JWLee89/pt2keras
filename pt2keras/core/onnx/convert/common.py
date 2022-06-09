import logging
import typing as t
from functools import wraps

import numpy as np

from pt2keras.core.onnx.graph import Graph, OnnxNode, TestResults

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
    if output_keras_layer is None:
        if node.op_type == 'Constant':
            _LOGGER.debug(f'Skipping test for Constant node: {node}')
        else:
            _LOGGER.warning(f'Output keras layer not available for: {node}. '
                            f'Skipping test.')
        return False

    # Print metadata during debug mode
    print(f'Input node count: {len(inputs)}, operation: {node.op_type}')
    for i, input in enumerate(inputs):
        if isinstance(input, np.ndarray):
            print(f'Input no. {i + 1} - {input.shape}')
        else:
            print(f'Input no. {i + 1} - {input}')

    # Perform inference with keras
    input_data = np.random.randn(*input_keras_layer.shape)
    keras_output = output_keras_layer(input_data).numpy()
    # Convert Keras output to original PyTorch shape if 4D output
    if len(keras_output.shape) == 4:
        keras_output = keras_output.transpose(0, 3, 1, 2)

    # Create onnx computational graph

    # Infer with onnx

    # Check whether the outputs are equal. If not, we need to

    print(f'Inference: {keras_output.shape}')
    print(f'Infered with node: {node}')

    # Create intermediate computational graph for inference
    # node = helper.make_node(node.op_type, inputs=input_nodes, outputs=['yee'],
    #                         value=helper.make_tensor(name='test_temp',
    #                         data_type = tp.FLOAT,dims = training_results[‘intercept’].shape,
    #                         vals = training_results[‘intercept’].flatten())

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
            _LOGGER.debug(f'Converting: {onnx_op}')
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
                    print(f'keras layer: {keras_tensor}')

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
