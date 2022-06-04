"""
All convolution operation converter
"""
import torch.nn as nn
from tensorflow import keras

from .common import converter


@converter(nn.Dropout)
def dropout(layer: nn.Dropout):
    return keras.layers.Dropout(layer.p)
