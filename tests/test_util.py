"""
Test the following file:

pts2keras/core/onnx/convert/util.py
"""
import typing as t

import numpy as np
import pytest

from pt2keras.core.onnx.util import keras_input_to_pt_shape, pt_input_to_keras_shape


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
    actual: t.Tuple = keras_input_to_pt_shape(test_input)
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
    actual: t.Tuple = pt_input_to_keras_shape(test_input)
    assert actual == expected, f'Expected: {expected}, actual: {actual}'
