import torch
import torch.nn as nn
import onnx
import tensorflow as tf

from pt2keras.core.onnx.graph import Graph
from pt2keras.core.onnx.util import get_graph_output_shape
from pt2keras.main import Pt2Keras


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # These can all be found using named_modules() or children()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, (1, 1), (2, 2), bias=True),
            nn.Conv2d(16, 32, (1, 1), (1, 1), bias=False),
        )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, 16, (1, 1), (2, 2), bias=True),
        #     # nn.BatchNorm2d(16),
        #     # nn.ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2))
        # )

    def forward(self, X):
        # we can retrieve these operations via .modules(), named_modules(), children(), etc.
        output = self.conv(X)
        output = (output + 6 * 3) / 3
        return output
        # return output[:, :, 2, 4]


def analyze(model):
    for name, module in model.named_children():
        if not list(module.children()) == []:  # if not leaf node, ski[
            print(f'has children: {name}')
        print(f'Name: {name}, module: {module}')


if __name__ == '__main__':
    model = Model().eval()

    analyze(model)
    x = torch.ones(1, 3, 224, 224)

    # Set the model to training mode to get the full computational graph
    # import torch._C as _C
    # TrainingMode = _C._onnx.TrainingMode

    torch.onnx.export(model,
                      x,
                      'test_model.onnx',
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=False,
                      # training=TrainingMode.TRAINING,
                      export_params=True,
                      input_names=['input_0'],
                      output_names=['output_0'])

    # # Load onnx model
    onnx_model = onnx.load_model('test_model.onnx')
    converter = Pt2Keras(onnx_model)
    # graph = Graph(onnx_model)
    # keras_model = graph.convert()
    output = get_graph_output_shape(onnx_model.graph, (0, 2, 3, 1))

    keras_model = converter.convert(model)
    #
#    pt_output = model(x).permute(0, 2, 3, 1)
    pt_output = model(x)
    if len(pt_output.shape) == 4:
        pt_output = pt_output.permute(0, 2, 3, 1)

    x_tf = tf.ones((1, 224, 224, 3))

    print(f'pytorch: {pt_output}')
    keras_output = keras_model(x_tf)
    print(f'keras: {keras_output}')

    keras_model.save('model.h5')
