import logging
import os.path
import typing as t
import onnx
import torch.nn as nn

from collections import OrderedDict

import torch.onnx
from onnx.helper import printable_node

from onnx import numpy_helper
from onnx.helper import get_attribute_value
from tensorflow import keras

from .util import get_graph_input_shape, test_model_output


class Graph:
    _SUPPORTED_OPERATIONS = {}
    _LOGGER = logging.getLogger('onnx::Graph')

    """
    A class that encapsulates and converts the onnx graph representation into
    Keras format
    """

    def __init__(self,
                 pytorch_model: nn.Module,
                 input_shape: t.Tuple,
                 opset_version: int = 13):
        """
        By default the onnx graph is designed to convert PyTorch onnx
        models to Keras. By making small modifications and writing converters,
        we can port this to work with other frameworks.

        For now, since I am working with computer vision models, this onnx converter
        is designed to work with models in the computer vision fields, that generally
        have at most, 4 dimensional inputs
        """
        self.pytorch_model = pytorch_model
        self.pytorch_input_shape = input_shape
        self.opset_version = opset_version

        # Added as class object for unit testing
        self.test_stats = None

        dummy_input = torch.randn(input_shape)
        output = self.pytorch_model(dummy_input)
        self.output_names = []
        if isinstance(output, (t.Tuple, t.List)):
            for i in range(len(output)):
                self.output_names.append(f'output_{i}')
        else:
            self.output_names.append('output_0')

        hash_str = f'__{hash(pytorch_model)}__.onnx'
        torch.onnx.export(self.pytorch_model,
                          dummy_input,
                          hash_str,
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          verbose=False,
                          opset_version=self.opset_version,
                          # training=TrainingMode.TRAINING,
                          export_params=True,
                          input_names=['input_0'],
                          output_names=self.output_names)

        # Check model
        self.onnx_model = onnx.load(hash_str)
        onnx.checker.check_model(self.onnx_model)
        if os.path.exists(hash_str):
            os.remove(hash_str)

        self.weights = OrderedDict()
        self.node_dict = OrderedDict()
        # The computational graph value we are building up
        self.computational_graph = {}

        # initialization phase:
        # ------------------------------------------------------

        # 1. Do forward pass over graphs to build up graph metadata
        self._init_graph()

        # 2. Initialize weight vectors from onnx graph
        self._initialize_weights()

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
        Forward pass over graph
        """
        for node in self.onnx_model.graph.node:
            key = node.name
            onnx_node_obj = OnnxNode(node)
            self.node_dict[key] = onnx_node_obj

            # Print node information
            node_info = printable_node(node)
            self._LOGGER.debug(f'NODE::::: {node_info}')

            if node.attribute:
                for attribute in node.attribute:
                    # INTS, FLOATS, etc
                    type_data = get_attribute_value(attribute)
                    # Try casting to numpy if possible
                    try:
                        data = numpy_helper.to_array(type_data)
                        onnx_node_obj.attributes[attribute.name] = data
                    # just save raw representation
                    except:
                        onnx_node_obj.attributes[attribute.name] = type_data

            # Constant nodes have no input and just single output
            # we need to extract constants
            if node.op_type == 'Constant':
                if len(node.output) != 1:
                    raise ValueError('Constant node does not have single output')
                # Get the raw output value
                for constant_node_output in node.output:
                    if constant_node_output not in self.computational_graph:
                        self.computational_graph[constant_node_output] = onnx_node_obj.attributes['value']

    def _convert(self):
        input_shape = self.pytorch_input_shape

        # Change image shape from BCHW to BHWC (TensorFlow / Keras default shape)
        dims = (i for i in range(len(input_shape))) if len(input_shape) != 4 else (0, 2, 3, 1)

        # For now, assume we are working with models that have a single input.
        # we will later need to test this with models that have multiple inputs
        input_shape = get_graph_input_shape(self.onnx_model.graph, dims)[0]['shape']

        # Create input object to feed to the model.
        # This will need to change in the future if we want to support multiple inputs
        inputs = keras.Input(batch_shape=input_shape)
        outputs = inputs

        # Store unsupported operations in a set so that
        # we can figure out what we operations we need to add in the future
        has_unsupported_ops = False
        unsupported_ops = set()
        output_list = []

        # Create a new test stats object for each conversion run.
        self.test_stats = TestResults()

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
            if has_unsupported_ops:
                continue

            conversion_func = self._SUPPORTED_OPERATIONS[op_type]
            node_inputs = []
            for input_node in node.input_nodes:
                if input_node in self.computational_graph:
                    node_inputs.append(self.computational_graph[input_node])

            # Convert to keras
            outputs = conversion_func(node, outputs, self.computational_graph,
                                      self.node_dict, self.test_stats, *node_inputs)

            # Add to output data if output_node found
            if node.output_nodes[0] in self.output_names:
                print(f'output node: {node.output_nodes}, value: {outputs}')
                output_list.append(outputs)

            Graph._LOGGER.info(f'Successfully converted: {node}')

        # Print model conversion result if needed
        self._LOGGER.info('Model conversion report: ###################### \n'
                          f'{self.test_stats.get_test_results()}')

        if has_unsupported_ops:
            unsupported_operations = "\n- ".join(unsupported_ops)
            raise ValueError('Failed to convert model. The following operations are currently unsupported: '
                             f'{unsupported_operations}')

        if len(output_list) > 1:
            outputs = output_list

        model = keras.Model(inputs, outputs)
        # Test the Keras model output.
        # Error will be asserted if the output dimensions or values are very different.
        test_model_output(self.pytorch_model, model, self.pytorch_input_shape, input_shape)
        return model

    def _initialize_weights(self):
        for weight in self.onnx_model.graph.initializer:
            name = weight.name
            np_weights = numpy_helper.to_array(weight)
            # if len(np_weights.shape) == len(self.source_format):
            #     Graph._LOGGER.info(f'Transposing weights: {name}')
            #     # H,W,IC,OC
            #     np_weights = np_weights.transpose([2, 3, 1, 0])

            # Note tht we can also check whether the initialize is actually a weight
            # or a constant by checking what the name endswith
            # is_weight = name.endswith('weight') or name.endswith('bias')

            # operations such as the division in
            # (output + 6 * 3) / 3
            # can be considered an initializer
            # in this case, we need to add a constant node to the graph
            # if not is_weight:

            # IMPORTANT: Note that computational graph also contains
            # not only weights, but constant values from constant nodes such as
            # when we do element-wise additional to a constant
            self.computational_graph[name] = np_weights

            self.weights[name] = {
                'weights': np_weights
            }

        # move Weights to node for convenience
        for node in self.node_dict.values():
            for input_node in node.input_nodes:
                if input_node in self.weights:
                    node.weights.append(self.weights[input_node]['weights'])

        # Remove weights dictionary, since it is no longer needed
        del self.weights

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

        return f'Total node count: {node_count}. \n' \
               f'Test nodes: {tested_nodes} \n' \
               f'Untested nodes: {untested_nodes}. \n' \
               f'Tested operations: {tested_operations} \n' \
               f'Untested operations: {untested_operations} \n'


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
        self.attributes = {}
        # This is for unit testing
        # Access to original ProtoBuf attributes
        self.original_attribute = node.attribute

    def __repr__(self):
        return f'name: {self.name}: ' \
               f'\n\t op type: {self.op_type}' \
               f'\n\t input_nodes: {self.input_nodes}' \
               f'\n\t output_nodes: {self.output_nodes}' \
               f'\n\t attributes: {self.attributes}'
