import logging

import torch
import torch.nn as nn
from pt2keras.core.convert.activations import silu
from pt2keras.core.convert.conv import conv2d

from pt2keras import Pt2Keras
import tensorflow as tf

if __name__ == '__main__':

    conv = nn.Conv2d(3, 16, (1, 1), (2, 2), bias=True)
    act = nn.SiLU()
    layers = nn.Sequential(
        conv,
        act
    )

    converter = Pt2Keras()
    converter.set_logging_level(logging.DEBUG)

    x = torch.ones(1, 3, 32, 32)
    output_pt = layers(x).permute(0, 2, 3, 1)

    model = converter.convert(layers)
    x = tf.ones((1, 32, 32, 3))
    output = model(x)
    print(f'Output: {output_pt}')
    print(f'Output: {output}')

    # x = act(x)
    # print(x)
    #
    # import tensorflow as tf
    # y = tf.ones([4])
    #
    # from tensorflow.keras.activations import swish
    #
    # y = swish(y)
    # import numpy as np
    # np.testing.assert_allclose(x.numpy(), y.numpy(), rtol=1e-5, atol=0)
