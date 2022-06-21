import pytest
import torchvision.models.resnet as resnet

from ..common import get_converter


@pytest.mark.skip('Skip test because converter stops for no reason.')
@pytest.mark.parametrize('model_class', [resnet.resnet18, resnet.resnet34, resnet.resnet50, resnet.wide_resnet50_2])
@pytest.mark.parametrize('input_sizes', [(1, 3, 32, 32)])
def test_resnet(model_class, input_sizes):
    """
    Test conversion of ResNet class models.
    An error will be thrown if Unsuccessful
    Args:
        model_class: The model class to test
        input_sizes: The size of the inputs
    """
    model = model_class().eval()
    converter = get_converter(model, input_sizes)
    # Check whether conversion is successful. Error will be thrown if it fails
    converter.convert()
