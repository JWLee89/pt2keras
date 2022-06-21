import pytest
import torchvision.models.efficientnet as efficientnet

from tests.common import get_converter


@pytest.mark.parametrize(
    'model_class',
    [
        efficientnet.efficientnet_b0,
        efficientnet.efficientnet_b1,
        efficientnet.efficientnet_b2,
        efficientnet.efficientnet_b3,
        efficientnet.efficientnet_b4,
        efficientnet.efficientnet_b5,
        efficientnet.efficientnet_b6,
        efficientnet.efficientnet_b7,
    ],
)
@pytest.mark.parametrize('input_sizes', [(1, 3, 32, 16)])
def test_efficientnet(model_class, input_sizes):
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
