import logging
import typing as t
import onnx
import tensorflow as tf


from collections import OrderedDict
from onnx.helper import printable_node

from onnx import numpy_helper
from onnx.helper import get_attribute_value
from tensorflow import keras

from .util import get_graph_input_shape, get_graph_output_shape, generate_node_key, get_tensor_data


class Graph:

    _SUPPORTED_OPERATIONS = {}
    _LOGGER = logging.getLogger('onnx::Graph')

    """
    A class that encapsulates and converts the onnx graph representation into
    Keras format
    """
    def __init__(self,
                 onnx_model: onnx.ModelProto,
                 source_format: t.Tuple = ('B', 'C', 'H', 'W'),
                 target_format: t.Tuple = ('B', 'H', 'W', 'C')):
        """
        By default the onnx graph is designed to convert PyTorch onnx
        models to Keras. By making small modifications and writing converters,
        we can port this to work with other frameworks.

        For now, since I am working with computer vision models, this onnx converter
        is designed to work with models in the computer vision fields, that generally
        have at most, 4 dimensional inputs

        Args:
            onnx_model: The target onnx model to convert
            source_format: The source input format
            target_format:
        """
        self.onnx_model = onnx_model
        self.source_format = source_format
        self.target_format = target_format
        onnx.checker.check_model(onnx_model)

        self.weights = OrderedDict()
        self.node_dict = OrderedDict()

        self.transpose_matrix = self._get_transpose_matrix()

        # The computational graph value we are building up
        self.computational_graph = {}

        # initialization phase:
        # ------------------------------------------------------

        # 1. Do forward pass over graphs to build up graph metadata
        self._init_graph()

        # 2. Initialize weight vectors from onnx graph
        self._initialize_weights()

    def _get_transpose_matrix(self):
        if len(self.source_format) != len(self.target_format):
            raise ValueError('The dimensions of the source and target format must be equal')
        # for now, just hardcode
        transpose_matrix = [0, 2, 3, 1]
        return transpose_matrix

    def inspect(self) -> t.Tuple[t.List, t.List]:
        """
        Inspect the model to see whether the current model is convertible.
        Returns:
            A two-tuple of supported and unsupported operations
        """
        supported, unsupported = [], []
        for idx, node in enumerate(self.onnx_model.graph.node):
            if node.op_type in self.converters:
                supported.append(node.op_type)
            else:
                unsupported.append(node.op_type)
        return supported, unsupported

    def convert_op(self, op):
        if op in Graph._SUPPORTED_OPERATIONS:
            Graph._LOGGER.debug(f'Converting opertion: {op} ... ')
            conversion_function = Graph._SUPPORTED_OPERATIONS[op]
            keras_layer = conversion_function(op)
            return keras_layer
        else:
            raise ValueError(f'Operation: {op} is not supported ... ')

    def _init_graph(self):
        """
        Forward pass over graph
        """
        for node in self.onnx_model.graph.node:
            key = node.name
            onnx_node_obj = OnnxNode(node)
            self.node_dict[key] = onnx_node_obj
            node_info = printable_node(node)
            print(f'NODE::::: {node_info}')

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

        print(f'Computational graph: {self.computational_graph.keys()}')

    def _convert(self):
        input_shape = get_graph_input_shape(self.onnx_model.graph, (0, 2, 3, 1))[0]['shape']
        inputs = keras.Input(batch_shape=input_shape)
        outputs = inputs

        has_unsupported_ops = False
        unsupported_ops = set()

        for node_key, node in self.node_dict.items():
            print(f'key: {node_key}, val: {node}')
            print('-' * 50)
            op_type = node.op_type
            if op_type not in self._SUPPORTED_OPERATIONS:
                unsupported_ops.add(op_type)
                has_unsupported_ops = True

            # no need to bother converting
            if has_unsupported_ops:
                continue

            conversion_func = self._SUPPORTED_OPERATIONS[op_type]

            # convert
            outputs = conversion_func(node, self.computational_graph, outputs)

        if has_unsupported_ops:
            raise ValueError('Failed to convert model. The following operations are currently unsupported ... '
                             f'{", ".join(unsupported_ops)}')

        print(f'Graph: {self.computational_graph}')
        print(f'Input: {inputs}')
        print(f'Outputs: {outputs}')

        model = keras.Model(inputs, outputs)
        return model

    def _initialize_weights(self):
        for weight in self.onnx_model.graph.initializer:
            print(f'WEIGHT::::: {weight.name}')
            name = weight.name
            np_weights = numpy_helper.to_array(weight)
            if len(np_weights.shape) == len(self.source_format):
                np_weights = np_weights.transpose([2, 3, 1, 0])

            is_weight = name.endswith('weight') or name.endswith('bias')

            # operations such as the division in
            # (output + 6 * 3) / 3
            # can be considered an initializer
            # in this case, we need to add a constant node to the graph
            if not is_weight:
                self.computational_graph[name] = np_weights

            self.weights[name] = {
                'weights': np_weights
            }

        # move data to nodes
        for node in self.node_dict.values():
            for input_node in node.input_nodes:
                if input_node in self.weights:
                    node.weights.append(self.weights[input_node]['weights'])

        # Remove data
        self.weights = None

    def convert(self):
        """
        Convert the model into the desired representation format
        """
        model = keras.Sequential()
        for idx, node in self.node_dict.items():
            pass

        return model


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

    def __repr__(self):
        return f'name: {self.name}: ' \
               f'\n\t op type: {self.op_type}' \
               f'\n\t input_nodes: {self.input_nodes}' \
               f'\n\t output_nodes: {self.output_nodes}' \
               f'\n\t attributes: {self.attributes}'
