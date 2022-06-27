import pytest
import torchvision.models.resnet as resnet

from ..common import do_conversion


@pytest.mark.parametrize('model_class', [resnet.resnet18, resnet.resnet34, resnet.resnet50, resnet.wide_resnet50_2])
@pytest.mark.parametrize('input_sizes', [(1, 3, 64, 64)])
def test_resnet(converter, model_class, input_sizes):
    """
    Test conversion of ResNet class models.
    An error will be thrown if Unsuccessful
    Args:
        model_class: The model class to test
        input_sizes: The size of the inputs
    """
    do_conversion(converter, model_class, input_sizes, strict=True)
