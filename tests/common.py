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
        (1, 3, 224, 224),
        (1, 3, 32, 32),
    ]


def get_converter(model, input_shape):
    return Pt2Keras(model, input_shape)
