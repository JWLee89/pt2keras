import onnx
from tensorflow import keras

from .common import converter


@converter('GlobalAveragePool')
def global_average_pool(node: onnx.NodeProto, input_layer, input_tensor):
    # axis = node.attributes['axis']
    global_pool = keras.layers.GlobalAveragePooling2D()
    output = global_pool(input_layer)

    def target_layer(x):
        return keras.backend.expand_dims(x)

    lambda_layer1 = keras.layers.Lambda(target_layer)
    lambda_layer2 = keras.layers.Lambda(target_layer)

    output = lambda_layer1(output)  # double expand dims
    output = lambda_layer2(output)
    print(f'input shape: {input_tensor.shape}, output shape: {output.shape}')
    return output


@converter('MaxPool')
def max_pool(node: onnx.NodeProto, input_layer, input_tensor):

    attributes = node.attributes
    kernel_shape = attributes['kernel_shape']
    stride_shape = attributes['strides']
    pads = attributes['pads'] if 'pads' in attributes else [0, 0, 0, 0, 0, 0]
    pad = 'valid'

    if all([shape % 2 == 1 for shape in kernel_shape]) and \
            all([kernel_shape[i] // 2 == pads[i] for i in range(len(kernel_shape))]) and \
            all([shape == 1 for shape in stride_shape]):
        pad = 'same'
        print('Use `same` padding parameters.')
    else:
        print('Unable to use `same` padding. Add ZeroPadding2D layer to fix shapes.')
        padding_name = node.name + '_pad'
        if len(kernel_shape) == 2:
            padding = None

            if len(pads) == 2 and (pads[0] > 0 or pads[1] > 0):
                padding = (pads[0], pads[1])
            elif len(pads) == 4 and (pads[0] > 0 or pads[1] > 0 or pads[2] > 0 or pads[3] > 0):
                padding = ((pads[0], pads[2]), (pads[1], pads[3]))

            if padding is not None:
                padding_layer = keras.layers.ZeroPadding2D(
                    padding=padding,
                    name=padding_name
                )
                input_layer = padding_layer(input_layer)
        else:  # 3D padding
            padding_layer = keras.layers.ZeroPadding3D(
                padding=pads[:len(stride_shape)],
                name=padding_name
            )
            input_layer = padding_layer(input_layer)

    if len(kernel_shape) == 2:
        pooling = keras.layers.MaxPooling2D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
        )
    elif len(kernel_shape) == 3:
        pooling = keras.layers.MaxPooling3D(
            pool_size=kernel_shape,
            strides=stride_shape,
            padding=pad,
        )
    else:
        raise ValueError('Pooling operation must be performed on 2D or 3D objects')

    return pooling(input_layer)

