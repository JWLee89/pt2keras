"""
All convolution operation converter
"""
import torch.nn as nn
from tensorflow import keras

from .common import converter


@converter(nn.MaxPool2d)
def max_pool_2d(pytorch_max_pool: nn.MaxPool2d):
    stride = pytorch_max_pool.stride
    pool_size = pytorch_max_pool.kernel_size
    strides = (stride, stride) if isinstance(stride, tuple) else stride
    return keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)
