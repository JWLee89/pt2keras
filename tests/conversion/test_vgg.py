import pytest
import torchvision.models.vgg as vgg

from tests.common import do_conversion


@pytest.mark.parametrize('model_class', [vgg.vgg11, vgg.vgg13, vgg.vgg16, vgg.vgg19, vgg.vgg11_bn])
@pytest.mark.parametrize('input_sizes', [(1, 3, 224, 224)])
def test_vgg(converter, model_class, input_sizes):
    """
    Test conversion of EfficientNet class models.
    An error will be thrown if Unsuccessful
    Args:
        model_class: The model class to test
        input_sizes: The size of the inputs
    """
    do_conversion(converter, model_class, input_sizes, strict=True)
