import numpy as np

from onnx import helper
import onnx
import onnxruntime as ort

# session = ort.InferenceSession('__8792818407915__.onnx')
model = onnx.load_model('__8792818407915__.onnx')

session = ort.InferenceSession(model.SerializeToString())

print(f'Session: {session}')

dummy = np.ones((1, 3, 224, 112), dtype=np.float32)
results = session.run(None, {"input_0": dummy})

x1 = np.ones(100)
x2 = np.ones(100)
#
# onnx_model: onnx.ModelProto = onnx.load_model('__8792818407915__.onnx')
#
# graph = onnx_model.graph.node
# for node in graph:
#     print(f'Node: {node}')
#     break
#
# for init in onnx_model.graph.initializer:
#     is_weight_or_bias = init.name.endswith('bias') or init.name.endswith('weight')
#     if not is_weight_or_bias and init.data_type != 1:
#         print(f'yee: {init.name}')

from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto


# Create one input (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])
pads = helper.make_tensor_value_info('pads', TensorProto.FLOAT, [1, 4])

value = helper.make_tensor_value_info('value', AttributeProto.FLOAT, [1])


# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
    'Pad',                  # name
    ['X', 'pads', 'value'], # inputs
    ['Y'],                  # outputs
    mode='constant',        # attributes
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],        # nodes
    'test-model',      # name
    [X, pads, value],  # inputs
    [Y],               # outputs
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example')
#
# print('The model is:\n{}'.format(model_def))
# onnx.checker.check_model(model_def)
# print('The model is checked!')
#
# yee = ort.InferenceSession(model_def.SerializeToString())
# print(f'Yeee: {yee}')
#
print(f'Results: {results[0].shape}')


