import logging
import os.path
import typing as t
from collections import OrderedDict

import onnx
import onnxruntime as ort
import torch.nn as nn
import torch.onnx
from onnx import numpy_helper
from onnx.helper import printable_node
from tensorflow import keras

from .testing.utils import test_model_output
from .util import get_graph_output_info, get_graph_shape_info


class Graph:
    _SUPPORTED_OPERATIONS = {}
    _LOGGER = logging.getLogger('onnx::Graph')

    """
    A class that encapsulates and converts the onnx graph representation into
    Keras format
    """

    def __init__(self, opset_version: int = 13):
        """
        By default the onnx graph is designed to convert PyTorch onnx
        models to Keras. By making small modifications and writing converters,
        we can port this to work with other frameworks.

        For now, since I am working with computer vision models, this onnx converter
        is designed to work with models in the computer vision fields, that generally
        have at most, 4 dimensional inputs
        """
        self.opset_version = opset_version

        # Model variables. This will be initialized when load_model is called.
        self.model = None
        self.pytorch_input_shape = None
        self.output_names = None

        self.node_dict = OrderedDict()
        # The computational graph value we are building up
        self.computational_graph = {}
        # This is for accessing keras input cache
        self.forward_input_cache = {}

    def _set_onnx_rt_session(self, onnx_path: str):
        """
        Given the path to onnx model, create and set the onnx runtime session
        object property for the given Graph instance
        Args:
            onnx_path: The name of the onnx path
        """
        self.onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(self.onnx_model)
        self.ort_session = ort.InferenceSession(onnx_path)

    def _load_model(self, model: t.Union[nn.Module, str], input_shape: t.Tuple):
        """
        Given a model, load the model and create the onnx runtime session object
        Args:
            model: The model or onnx model path.
            input_shape: The shape of the input to be fed into the neural network.
        """
        self.model = model
        self.pytorch_input_shape = input_shape
        if isinstance(self.model, nn.Module):
            # For now, we assume that there is only a single input
            # We can later change this to support multiple inputs
            dummy_input = torch.randn(input_shape)
            output = self.model(dummy_input)

            # Parse output names
            self.output_names = []
            if isinstance(output, (t.Tuple, t.List)):
                for i in range(len(output)):
                    self.output_names.append(f'output_{i}')
            else:
                self.output_names.append('output_0')

            # Save PyTorch model as .onnx
            hash_str = f'__{hash(model)}__.onnx'
            torch.onnx.export(
                self.model,
                dummy_input,
                hash_str,
                do_constant_folding=True,  # whether to execute constant folding for optimization
                verbose=False,
                opset_version=self.opset_version,
                export_params=True,
                input_names=['input_0'],
                output_names=self.output_names,
            )

            # Check model and create onnx runtime session for inference
            self._set_onnx_rt_session(hash_str)
            if os.path.exists(hash_str):
                os.remove(hash_str)
        elif isinstance(self.model, str):
            self._set_onnx_rt_session(model)
        else:
            raise ValueError(
                f'Invalid model type: {self.model}. Please pass in ' f'PyTorch model or onnx model string path.'
            )

    def inspect(self) -> t.Tuple[t.List, t.List]:
        """
        Inspect the model to see whether the current model is convertible.
        Returns:
            A two-tuple of supported and unsupported operations
        """
        supported, unsupported = [], []
        for idx, node in enumerate(self.onnx_model.graph.node):
            if node.op_type in self._SUPPORTED_OPERATIONS:
                supported.append(node.op_type)
            else:
                unsupported.append(node.op_type)
        return supported, unsupported

    def set_logging_level(self, logging_level):
        self._LOGGER.setLevel(logging_level)

    def _init_graph(self):
        """
        Forward pass over graph to
        generate metadata
        """
        for node in self.onnx_model.graph.node:
            key = node.name
            onnx_node_obj = OnnxNode(node)
            self.node_dict[key] = onnx_node_obj

            # Print node information
            node_info = printable_node(node)
            self._LOGGER.debug(f'NODE::::: {node_info}')

            # Constant nodes have no input and just single output
            # we need to extract constants
            if node.op_type == 'Constant':
                if len(node.output) != 1:
                    raise ValueError('Constant node does not have single output')
                # Get the raw output value
                for constant_node_output in node.output:
                    if constant_node_output not in self.computational_graph:
                        self.computational_graph[constant_node_output] = onnx_node_obj.attributes['value']

    def convert(self, model: t.Union[nn.Module, str], input_shape: t.Tuple, strict: bool = False):
        """
        Convert the PyTorch model into Keras
        """
        # initialization phase:
        # ------------------------------------------------------

        # 1. Load the model
        self._load_model(model, input_shape)

        # 2. Do forward pass over graphs to build up graph metadata
        self._init_graph()

        # 3. Initialize weight vectors from onnx graph
        self._initialize_weights()

        # 4. Build keras model during second forward pass
        input_shape = self.pytorch_input_shape

        # Change image shape from BCHW to BHWC (TensorFlow / Keras default shape)
        dims = (i for i in range(len(input_shape))) if len(input_shape) != 4 or input_shape[-1] == 3 else (0, 2, 3, 1)

        # For now, assume we are working with models that have a single input.
        # we will later need to test this with models that have multiple inputs
        self.output_names = [output['name'] for output in get_graph_output_info(self.onnx_model.graph)]

        # For now, assume we are working with models that have a single input.
        # we will later need to test this with models that have multiple inputs
        input_data = get_graph_shape_info(self.onnx_model.graph, dims)[0]
        input_shape = input_data['shape']
        input_name = input_data['name']

        # First dimension = 0 is dynamic batching. We disallow those ...
        if input_shape[0] == 0:
            error_msg = 'Cannot convert model with dynamic batch size ... '
            self._LOGGER.error(error_msg)
            raise ValueError(error_msg)

        # Create input object to feed to the model.
        # This will need to change in the future if we want to support multiple inputs
        inputs = keras.Input(batch_shape=input_shape)
        self.forward_input_cache[input_name] = inputs
        outputs = inputs

        # Store unsupported operations in a set so that
        # we can figure out what we operations we need to add in the future
        has_unsupported_ops = False
        unsupported_ops = set()
        output_list = []

        # Create a new test stats object for each conversion run.
        test_stats = TestResults()

        # Convert the model
        for node_key, node in self.node_dict.items():
            op_type = node.op_type

            if op_type not in self._SUPPORTED_OPERATIONS:
                unsupported_ops.add(op_type)
                has_unsupported_ops = True

            # no need to bother converting if the library
            # does not have all the necessary converters.
            # Users can extend the library by adding the required support
            # without modifying the library directly
            # -----------------------------------------------
            # Constant nodes have no input
            if has_unsupported_ops or node.name.startswith('Constant'):
                continue

            conversion_func = self._SUPPORTED_OPERATIONS[op_type]
            node_inputs = []

            # add inputs
            for input_node in node.input_nodes:
                if input_node in self.computational_graph:
                    node_inputs.append(self.computational_graph[input_node])

            input_node_name = node.input_nodes[0]
            # retrieve previous layer
            if input_node_name in self.forward_input_cache:
                input_keras_layer = self.forward_input_cache[input_node_name]
            # Is a weight node
            else:
                input_keras_layer = self.computational_graph[input_node_name]

            # Convert to keras
            outputs = conversion_func(
                node, input_keras_layer, self.computational_graph, self.opset_version, test_stats, *node_inputs
            )

            self.forward_input_cache[node.name] = outputs

            # Add to output data if output_node found
            for output_node in node.output_nodes:
                if output_node in self.output_names:
                    output_list.append(outputs)

            Graph._LOGGER.info(f'Successfully converted: {node}')

        # Print model conversion result if needed
        self._LOGGER.info('Model conversion report: ###################### \n' f'{test_stats.get_test_results()}')

        if has_unsupported_ops:
            unsupported_operations = '\n- '.join(unsupported_ops)
            raise ValueError(
                'Failed to convert model. The following operations are currently unsupported: \n'
                f'- {unsupported_operations}'
            )

        # In case there are multiple outputs
        if len(output_list) > 1:
            outputs = output_list

        model = keras.Model(inputs, outputs)
        # Test the Keras model output.
        # Error will be asserted if the output dimensions or values are very different.
        test_model_output(self.model, model, self.pytorch_input_shape, strict)
        return model

    def _initialize_weights(self):
        for weight in self.onnx_model.graph.initializer:
            name = weight.name
            np_weights = numpy_helper.to_array(weight)

            # operations such as the division in
            # (output + 6 * 3) / 3
            # can be considered an initializer
            # in this case, we need to add a constant node to the graph
            # if not is_weight:

            # IMPORTANT: Note that computational graph also contains
            # not only weights, but constant values from constant nodes such as
            # when we do element-wise additional to a constant
            self.computational_graph[name] = np_weights

        # move Weights to node for convenience
        for node in self.node_dict.values():
            for input_node in node.input_nodes:
                if input_node in self.computational_graph:
                    node.weights.append(self.computational_graph[input_node])

        self._LOGGER.info(f'Built computational graph: {self.computational_graph.keys()}')


class TestResults:
    """
    The class contains test results when converting the model on a node-by-node basis
    """

    def __init__(self):
        self.tested_nodes: t.List = []
        self.untested_nodes: t.List = []

        self.tested_operations: t.Set = set()
        self.untested_operations: t.Set = set()

    def get_test_results(self, stat_delimiter: str = '\n - ') -> str:
        """
        Get a formatted string visualizing the test results.
        Args:
            stat_delimiter: The delimiter for each entry.
            The default format is '-' delimited bullet points.
            E.g.
            - point1
            - point2

        Returns:
            The formatted string containing the test results representing the
            conversion results of a target model.
        """
        # Node-related stats
        node_count = len(self.tested_nodes) + len(self.untested_nodes)
        tested_nodes = stat_delimiter + stat_delimiter.join(self.tested_nodes)
        untested_nodes = stat_delimiter + stat_delimiter.join(self.untested_nodes)

        # Operation states
        tested_operations = stat_delimiter + stat_delimiter.join(self.tested_operations)
        untested_operations = stat_delimiter + stat_delimiter.join(self.untested_operations)

        return (
            f'Total node count: {node_count}. \n'
            f'Test nodes: {tested_nodes} \n'
            f'Untested nodes: {untested_nodes}. \n'
            f'Tested operations: {tested_operations} \n'
            f'Untested operations: {untested_operations} \n'
        )


def onnx_node_attributes_to_dict(attributes):
    """
    From: https://github.com/gmalivenko/onnx2keras/blob/master/onnx2keras/converter.py
    Parse ONNX attributes to Python dictionary
    Args:
        attributes: The attributes from OnnxNode
    """

    def onnx_attribute_to_dict(onnx_attr):
        """
        Parse ONNX attribute
        Args:
            onnx_attr: ONNX attribute
        Returns:
            Python data type
        """
        if onnx_attr.HasField('t'):
            return numpy_helper.to_array(getattr(onnx_attr, 't'))

        for attr_type in ['f', 'i', 's']:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        for attr_type in ['floats', 'ints', 'strings']:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))

    return {arg.name: onnx_attribute_to_dict(arg) for arg in attributes}


class OnnxNode:
    """
    A node in the onnx graph
    """

    def __init__(self, node):
        self.name = node.name
        self.op_type = node.op_type
        self.input_nodes = node.input
        self.output_nodes = node.output
        self.weights = []
        self.attributes = onnx_node_attributes_to_dict(node.attribute)

    def __repr__(self):
        return (
            f'name: {self.name}: '
            f'\n\t op type: {self.op_type}'
            f'\n\t input_nodes: {self.input_nodes}'
            f'\n\t output_nodes: {self.output_nodes}'
            f'\n\t attributes: {self.attributes}'
        )
