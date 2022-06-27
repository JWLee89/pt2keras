import typing as t


def input_sizes_to_test():
    """
    Get common input sizes to run tests.
    Note that some networks demand that input height
    and width be equal, so common input sizes
    are made intentionally square. Feel free to
    also provide rectangular input as well.
    """
    return [
        (1, 3, 64, 64),
        (1, 3, 32, 32),
    ]


def do_conversion(converter, model_class: t.Any, input_sizes: t.Tuple, strict: bool = False) -> None:
    """
    A common function for running a conversion operation.
    This will be run for each type of model that is supported
    Args:
        converter: The Pt2keras instance
        model_class: The nn.Module class item
        input_sizes: A tuple specifying the dimension of the input
        strict: If set to true, even small deviations to the output between PyTorch vs Keras
        will result in an error being thrown
    """
    model = model_class().eval()
    # Check whether conversion is successful. Error will be thrown if it fails
    converter.convert(model, input_sizes, strict)
