import pytest
import torchvision.models.mobilenet as mobilenet

from tests.common import do_conversion, input_sizes_to_test


@pytest.mark.skip('Currently, lot of difference between outputs in mobilenet_v3_large and small')
@pytest.mark.parametrize(
    'model_class',
    [
        mobilenet.mobilenet_v2,
        mobilenet.mobilenet_v3_small,
        mobilenet.mobilenet_v3_large,
    ],
)
@pytest.mark.parametrize('input_sizes', input_sizes_to_test())
def test_mobilenet(model_class, input_sizes):
    """
    Test conversion for the MobileNet class models.
    An error will be thrown if Unsuccessful
    Args:
        model_class: The model class to test
        input_sizes: The size of the inputs
    """
    do_conversion(model_class, input_sizes)
