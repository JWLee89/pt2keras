"""
Test the following file:

pts2keras/core/onnx/convert/util.py
"""
import typing as t

import numpy as np
import pytest

from src.pt2keras.onnx_backend.util import keras_input_to_pt, pt_input_to_keras


@pytest.mark.parametrize(
    'test_input,expected',
    [
        (
            np.random.randn(2, 32, 16, 3),
            (2, 3, 32, 16),
        ),
        (
            np.random.randn(6, 65, 24, 3),
            (6, 3, 65, 24),
        ),
    ],
)
def test_keras_input_to_pt_shape(test_input: np.ndarray, expected: t.Tuple):
    actual: t.Tuple = keras_input_to_pt(test_input).shape
    assert actual == expected, f'Expected: {expected}, actual: {actual}'


@pytest.mark.parametrize(
    'test_input,expected',
    [
        (
            np.random.randn(2, 3, 32, 16),
            (2, 32, 16, 3),
        ),
        (np.random.randn(9, 10, 5, 4, 3, 2), (9, 5, 4, 3, 2, 10)),
    ],
)
def test_pt_input_to_keras_shape(test_input: np.ndarray, expected: t.Tuple):
    actual = pt_input_to_keras(test_input).shape
    assert actual == expected, f'Expected: {expected}, actual: {actual}'
