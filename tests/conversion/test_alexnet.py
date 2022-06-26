import pytest
from torchvision.models.alexnet import alexnet

from tests.common import do_conversion


@pytest.mark.parametrize('input_sizes', [(1, 3, 224, 224)])
def test_vgg(input_sizes):
    """
    Test conversion of EfficientNet class models.
    An error will be thrown if Unsuccessful
    Args:
        input_sizes: The size of the inputs
    """
    do_conversion(alexnet, input_sizes)
