from .input_generator import *
import numpy as np


def test_conversion():
    a = np.array([1.5, 3.14], np.float32)
    print(a)
    b = [convert_f32_to_bf16(x) for x in a]
    print(b)
    c = [convert_bf16_to_f32(x) for x in b]
    print(c)
    assert c[0] == 1.5
    assert c[1] == 3.125
