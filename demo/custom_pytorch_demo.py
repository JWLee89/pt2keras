"""
To run the script, type in the following
python custom_pytorch_demo.py --input_shape 1 3 224 224
"""
from copy import deepcopy

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from common import default_args

from pt2keras import Pt2Keras


class DummyModel(nn.Module):
    """
    Model will be converted to EdgeTPU.
    """

    def __init__(self):
        super().__init__()
        # These can all be found using named_modules() or children()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), stride=(1, 1), padding=1, groups=1, bias=True),
            nn.Sigmoid(),
            # Downsample
            nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=1, dilation=(1, 1), groups=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(64, 128, (1, 1), stride=(2, 2), padding=1, groups=1, dilation=(1, 1), bias=False),
            # nn.Conv2d(3, 32, (3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(128, 256, (3, 3), stride=(2, 2), padding=2, groups=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(256, 512, (3, 3), stride=(2, 2), padding=2, groups=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(512, 1024, (2, 2), stride=(2, 2), padding=0, groups=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(1024, 256, (2, 2), stride=(1, 1), padding=0, groups=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(256, 128, (2, 2), stride=(1, 1), padding=0, groups=1, bias=True),
        )

    def forward(self, X):
        output = self.conv(X)
        output = torch.flatten(output, start_dim=1)
        return output


if __name__ == '__main__':
    args = default_args()
    shape = args.input_shape
    model = DummyModel()

    converter = Pt2Keras()

    x_pt = torch.randn(shape)
    # Generate dummy inputs
    x_keras = tf.convert_to_tensor(deepcopy(x_pt.numpy()))

    # input dimensions for PyTorch are BCHW, whereas TF / Keras default is BHWC
    if len(x_keras.shape) == 4:
        x_keras = tf.transpose(x_keras, (0, 2, 3, 1))

    print(f'pt shape: {x_pt.shape}, x_keras.shape: {x_keras.shape}')

    keras_model: tf.keras.Model = converter.convert(model, shape)

    # Make PT model the same input dimension as Keras
    # If the output is >= 4 dimensional
    pt_output = model(x_pt).cpu().detach().numpy()
    keras_output = keras_model(x_keras).numpy()
    if len(keras_output.shape) == 4:
        keras_output = keras_output.transpose(0, 3, 1, 2)
    # Mean average diff over all axis

    average_diff = np.mean(pt_output - keras_output)
    print(f'pytorch: {pt_output.shape}')
    print(f'keras: {keras_output.shape}')
    # The differences will be precision errors from multiplication / division
    print(f'Mean average diff: {average_diff}')
    print(f'Pytorch output tensor: {pt_output}')
    print(f'Keras output tensor: {keras_output}')
