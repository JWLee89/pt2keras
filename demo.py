import torch.nn as nn
from pt2keras.core.convert.activations import silu

if __name__ == '__main__':
    conv = nn.Conv2d(3, 16, (1, 1), (2, 2), bias=True)
    act = nn.SiLU()
    import torch


    x = torch.ones([4])

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
