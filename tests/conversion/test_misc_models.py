import pytest
from torchvision.models.alexnet import alexnet
from torchvision.models.googlenet import googlenet
from torchvision.models.inception import inception_v3

from tests.common import do_conversion


@pytest.mark.parametrize(
    'model_class',
    [
        alexnet,
        # Inception_v3 has high difference values.
        # Thus, we will set strict=False to pass test
        inception_v3,
        googlenet,
    ],
)
@pytest.mark.parametrize('input_sizes', [(1, 3, 224, 224)])
def test_vgg(converter, model_class, input_sizes):
    """
    Test conversion of EfficientNet class models.
    An error will be thrown if Unsuccessful
    Args:
        input_sizes: The size of the inputs
    """
    do_conversion(converter, model_class, input_sizes, strict=False)
