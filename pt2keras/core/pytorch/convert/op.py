import tensorflow as tf
import torch.nn as nn
from tensorflow import keras

import torchvision

from .common import converter


@converter(torchvision.ops.StochasticDepth)
def stochastic_depth(layer):
    # Does nothing when model is eval mode
    return keras.layers.Layer()
