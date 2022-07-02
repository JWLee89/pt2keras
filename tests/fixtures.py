import pytest

from src.pt2keras import Pt2Keras


@pytest.fixture
def converter() -> Pt2Keras:
    """
    Used in all conversion instances to avoid duplicate instances
    of instantiations and to prevent the case where multiple files need to be updated
    simultaneously:

    Returns:
        Instance

    """
    return Pt2Keras()
