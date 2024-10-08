from .input_generator import *
import numpy as np


def test_conversion():
    """
    Check that float(bfloat(a)) is (almost) a.
    """
    expected = np.array([[1.5, 3.125, -1.5], [-32, 0, -3.125]], dtype=np.float32)
    a = np.array([[1.5, 3.14, -1.5], [-32, 0, -3.14]], np.float32)
    b = f32_to_bf16(a)
    c = bf16_to_f32(b)
    assert np.allclose(c, expected, 0, 0)
