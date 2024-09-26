import random

import numpy as np
import pytest

from iree.compiler.dialects import arith, tensor, linalg
from iree.compiler.dialects.arith import _is_float_type
from iree.compiler.dialects.func import func
from iree.compiler.extras import types as T
from .conftest import invokable_module, mlir_type_to_np_dtype


def test_smol_matmul(session_module):
    session, module = session_module

    @func(T.tensor(32, 16, T.i8()), T.tensor(16, 32, T.i8()))
    def matmul_i8_i32(lhs, rhs):
        cst = arith.constant(T.i32(), 0)
        v0 = tensor.empty([32, 32], T.i32())
        v1 = linalg.fill(cst, outs=[v0])
        return linalg.matmul(lhs, rhs, outs=[v1])

    arg0 = np.ones((32, 16), dtype=np.int8)
    arg1 = np.ones((16, 32), dtype=np.int8)
    with invokable_module(session, module) as module:
        results = module[matmul_i8_i32.__name__](arg0, arg1).to_host()
        assert np.array_equal(results, arg0 @ arg1)


def emit_matmul(M, K, N, lhs_rhs_type, acc_type):
    matmul_name = f"{M}x{K}x{N}x{lhs_rhs_type}x{acc_type}"

    init_value = 0
    if _is_float_type(acc_type):
        init_value = 0.0

    @func(T.tensor(M, K, lhs_rhs_type), T.tensor(K, N, lhs_rhs_type), name=matmul_name)
    def matmul(lhs, rhs):
        cst = arith.constant(acc_type, init_value)
        v0 = tensor.empty([M, N], acc_type)
        v1 = linalg.fill(cst, outs=[v0])
        return linalg.matmul(lhs, rhs, outs=[v1])

    return matmul_name


testdata = [
    (32, 16, 32, T.i8, T.i32),
    (32, 32, 32, T.i8, T.i32),
    (64, 32, 64, T.i8, T.i32),
    (64, 64, 64, T.i8, T.i32),
    (128, 64, 128, T.i8, T.i32),
    (128, 128, 128, T.i8, T.i32),
    (128, 256, 128, T.i8, T.i32),
    (32, 16, 32, T.f32, T.f32),
    (32, 32, 32, T.f32, T.f32),
    (64, 32, 64, T.f32, T.f32),
    (64, 64, 64, T.f32, T.f32),
    (128, 128, 128, T.f32, T.f32),
    (128, 256, 128, T.f32, T.f32),

    (32, 16, 32, T.i8, T.i32),
    (32, 32, 32, T.i8, T.i32),
    (64, 32, 64, T.i8, T.i32),
    (64, 64, 64, T.i8, T.i32),
    (128, 64, 128, T.i8, T.i32),
    (128, 128, 128, T.i8, T.i32),
    (128, 256, 128, T.i8, T.i32),
    (32, 16, 32, T.f32, T.f32),
    (32, 32, 32, T.f32, T.f32),
    (64, 32, 64, T.f32, T.f32),
    (64, 64, 64, T.f32, T.f32),
    (128, 128, 128, T.f32, T.f32),
    (128, 256, 128, T.f32, T.f32),

    (32, 16, 32, T.i8, T.i32),
    (32, 32, 32, T.i8, T.i32),
    (64, 32, 64, T.i8, T.i32),
    (64, 64, 64, T.i8, T.i32),
    (128, 64, 128, T.i8, T.i32),
    (128, 128, 128, T.i8, T.i32),
    (128, 256, 128, T.i8, T.i32),
    (32, 16, 32, T.f32, T.f32),
    (32, 32, 32, T.f32, T.f32),
    (64, 32, 64, T.f32, T.f32),
    (64, 64, 64, T.f32, T.f32),
    (128, 128, 128, T.f32, T.f32),
    (128, 256, 128, T.f32, T.f32),
]


random.shuffle(testdata)

def ids(datum):
    if callable(datum):
        return datum.__name__
    return datum


@pytest.mark.parametrize("M, K, N, lhs_rhs_type, acc_type", testdata, ids=ids)
def test_matmul(session_module, M, K, N, lhs_rhs_type, acc_type):
    session, module = session_module

    lhs_rhs_type, acc_type = lhs_rhs_type(), acc_type()
    matmul_name = emit_matmul(M, K, N, lhs_rhs_type, acc_type)

    lhs_rhs_type = mlir_type_to_np_dtype(lhs_rhs_type)
    acc_type = mlir_type_to_np_dtype(acc_type)
    arg0 = np.ones((M, K), dtype=lhs_rhs_type)
    arg1 = np.ones((K, N), dtype=lhs_rhs_type)
    with invokable_module(session, module) as module:
        results = module[matmul_name](arg0, arg1).to_host()
        assert np.array_equal(results, (arg0.astype(acc_type) @ arg1.astype(acc_type)))
