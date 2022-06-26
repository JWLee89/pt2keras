import typing as t

from pt2keras import Pt2Keras


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


def get_converter():
    return Pt2Keras()


def do_conversion(model_class: t.Any, input_sizes: t.Tuple):
    model = model_class().eval()
    converter = get_converter()
    # Check whether conversion is successful. Error will be thrown if it fails
    converter.convert(model, input_sizes)
