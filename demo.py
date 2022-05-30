import logging

import torch
import torch.nn as nn

from pt2keras import Pt2Keras
import tensorflow as tf

from pt2keras.core.convert.common import converter


@converter(nn.MaxPool2d, override=True)
def max_pool_2d(pytorch_max_pool: nn.MaxPool2d):
    stride = pytorch_max_pool.stride
    pool_size = pytorch_max_pool.kernel_size
    strides = (stride, stride) if isinstance(stride, tuple) else stride
    return tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)


if __name__ == '__main__':
    conv = nn.Conv2d(3, 16, (1, 1), (2, 2), bias=True)
    act = nn.SiLU()
    conv_transpose = nn.ConvTranspose2d(16, 32, kernel_size=(2, 2), stride=(2, 2))
    layers = nn.Sequential(
        conv,
        act,
        nn.MaxPool2d(2),
        conv_transpose,
    )

    converter = Pt2Keras()
    converter.set_logging_level(logging.DEBUG)

    from torchvision.models.efficientnet import efficientnet_b0
    efficientnet = efficientnet_b0(pretrained=True)
    supported_layers, unsupported_layers = converter.inspect(efficientnet)
    print(f'Supported layers: {supported_layers}')
    print(f'Unsupported layers: {unsupported_layers}')
    print(f'Is convertible: {converter.is_convertible(layers)}')

    x = torch.ones(1, 3, 32, 32)
    output_pt = layers(x).permute(0, 2, 3, 1)

    model = converter.convert(layers)
    x = tf.ones((1, 32, 32, 3))
    output = model(x)
    print(f'Output: {output_pt.shape}')
    print(f'Output: {output.shape}')
    # print(f'Output: {output_pt}')
    # print(f'Output: {output}')
