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
            source_format: The source input format
            target_format:
        """
        self.pytorch_model = pytorch_model
        self.pytorch_input_shape = input_shape
        self.source_format = source_format
        self.target_format = target_format

        dummy_input = torch.randn(input_shape)
        output = self.pytorch_model(dummy_input)
        output_names = []
        if isinstance(output, (t.Tuple, t.List)):
            for i in range(len(output)):
                output_names.append(f'output_{i}')
        else:
            output_names.append('output_0')

        hash_str = f'__{hash(pytorch_model)}__.onnx'
        torch.onnx.export(self.pytorch_model,
                          dummy_input,
                          hash_str,
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          verbose=False,
                          # training=TrainingMode.TRAINING,
                          export_params=True,
                          input_names=['input_0'],
                          output_names=output_names)

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

    def _convert(self):
        input_shape = get_graph_input_shape(self.onnx_model.graph, (0, 2, 3, 1))[0]['shape']
        inputs = keras.Input(batch_shape=input_shape)
        outputs = inputs

        has_unsupported_ops = False
        unsupported_ops = set()

        # Convert the model
        for node_key, node in self.node_dict.items():
            op_type = node.op_type
            if op_type not in self._SUPPORTED_OPERATIONS:
                unsupported_ops.add(op_type)
                has_unsupported_ops = True

            # no need to bother converting
            if has_unsupported_ops:
                continue

            conversion_func = self._SUPPORTED_OPERATIONS[op_type]
            node_inputs = []
            for input_node in node.input_nodes:
                if input_node in self.computational_graph:
                    node_inputs.append(self.computational_graph[input_node])


            # convert to keras
            print(f'Attempting to convert node: {node}')
            print(f'Computational graph: {self.computational_graph.keys()}')
            outputs = conversion_func(node, outputs, self.computational_graph, *node_inputs)
            print(f'Successfully converted: {node} to {outputs}')

        if has_unsupported_ops:
            raise ValueError('Failed to convert model. The following operations are currently unsupported ... '
                             f'{", ".join(unsupported_ops)}')

        model = keras.Model(inputs, outputs)
        test_model_output(self.pytorch_model, model, self.pytorch_input_shape, input_shape)
        return model

    def _initialize_weights(self):
        for weight in self.onnx_model.graph.initializer:
            name = weight.name
            np_weights = numpy_helper.to_array(weight)
            if len(np_weights.shape) == len(self.source_format):
                print(f'transposing weight: {weight.name}')
                # H,W,IC,OC
                np_weights = np_weights.transpose([2, 3, 1, 0])
                print(f'np_weights: {np_weights.shape}')


            # is_weight = name.endswith('weight') or name.endswith('bias')

            # operations such as the division in
            # (output + 6 * 3) / 3
            # can be considered an initializer
            # in this case, we need to add a constant node to the graph
            # if not is_weight:
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
        del self.weights

        print(f'Computational grpah: {self.computational_graph.keys()}')


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
