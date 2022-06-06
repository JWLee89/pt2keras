import logging
import typing as t

from functools import wraps

from pt2keras.core.onnx.graph import Graph

_LOGGER = logging.getLogger('onnx:converter')


class DuplicateOperatorConverterError(Exception):
    pass


def converter(onnx_op: str,
              override: bool = False):
    """

    Args:
        onnx_op: An onnx operation that we wish to add a converter for
        override: If true, will override existing converter if it exists.
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
        def created_converter(onnx_node, output_layer, computational_graph, *args, **kwargs) -> t.Any:
            """
            Given a pytorch operation or layer, directly port it to keras
            Returns:
                The converted keras layer with all states copied / patched onto the keras layer
            """

            # Should add all available arguments and so on
            keras_layer = wrapped_fn(onnx_node, output_layer, *args, **kwargs)

            # build computational graph
            for output_node_name in onnx_node.output_nodes:
                if output_node_name not in computational_graph:
                    computational_graph[output_node_name] = keras_layer

            # Post processing
            # -------------------

            # 1. Perform tests to see whether the two layers (pytorch and keras)
            # are outputting the same value and shape
            # test_layer = _test_layer if output_testing_fn is None else output_testing_fn
            #
            # # try testing. If it fails, try adding custom test via decorator
            # try:
            #     test_layer(pytorch_layer, keras_layer, *args, **kwargs)
            # except:
            #     # If passed in test layers, we will skip the test.
            #     # Instead, a warning will be issued.
            #     _LOGGER.warning(f'Test failed for layer: {pytorch_module}. Skipping ... ')

            return keras_layer

        if not override:
            _LOGGER.info(f'Registering onnx node converter: {onnx_op}')
        Graph._SUPPORTED_OPERATIONS[onnx_op] = created_converter
        return created_converter

    return inner
