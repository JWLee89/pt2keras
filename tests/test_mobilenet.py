import pytest
import torchvision.models.mobilenet as mobilenet

from .common import get_converter, input_sizes_to_test


@pytest.mark.skip('HardSigmoid currently not supported.')
@pytest.mark.parametrize('model_class', [
    mobilenet.mobilenet_v2,
    mobilenet.mobilenet_v3_small,
    mobilenet.mobilenet_v3_large,
])
@pytest.mark.parametrize('input_sizes', input_sizes_to_test())
def test_mobilenet(model_class, input_sizes):
    """
    Test conversion for the MobileNet class models.
    An error will be thrown if Unsuccessful
    Args:
        model_class: The model class to test
        input_sizes: The size of the inputs
    """
    model = model_class().eval()
    converter = get_converter(model, input_sizes)
    # Check whether conversion is successful. Error will be thrown if it fails
    converter.convert()
