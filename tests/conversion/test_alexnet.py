import pytest
from torchvision.models.alexnet import alexnet

from tests.common import get_converter


@pytest.mark.parametrize('model_class', [alexnet])
@pytest.mark.parametrize('input_sizes', [(1, 3, 224, 224)])
def test_vgg(model_class, input_sizes):
    """
    Test conversion of EfficientNet class models.
    An error will be thrown if Unsuccessful
    Args:
        model_class: The model class to test
        input_sizes: The size of the inputs
    """
    model = model_class().eval()
    converter = get_converter(model, input_sizes)
    # Check whether conversion is successful. Error will be thrown if it fails
    converter.convert()
