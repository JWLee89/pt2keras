from . import converter


@converter(nn.SiLU(), keras.activations.swish)
def silu(pytorch_layer):
    """
        Given a PyTorch conv2d layer, output the equivalent keras conversion
        Args:
            pytorch_conv2d: The conv2d layer to convert

        Returns:
            The converted conv2d layer
        """
    # Add Stride
    print('yee')
    keras_layer = keras.activations.swish()
    return keras_layer
