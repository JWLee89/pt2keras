import pytest
import torchvision.models.resnet as resnet
from pt2keras import Pt2Keras


def input_sizes_to_test():
    return [
        (1, 3, 224, 224),
        (1, 3, 112, 112),
    ]


def get_converter(model, input_shape):
    return Pt2Keras(model, input_shape)


@pytest.mark.parametrize('model_class', [
    resnet.resnet18,
    resnet.resnet34,
    resnet.resnet50,
    resnet.resnet101,
])
@pytest.mark.parametrize('input_sizes', input_sizes_to_test())
def test_resnet(model_class, input_sizes):
    """
    Test conversion of ResNet class models.
    An error will be thrown if Unsuccessful
    Args:
        model_class: The
        input_sizes:

    Returns:

    """
    model = model_class().eval()
    converter = get_converter(model, input_sizes)
    converter.convert()
