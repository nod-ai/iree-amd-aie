# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deterministic regression test for the amdxdna per-binding `byte_offset`
propagation bug.

`byte_offset_propagation.mlir` is a 2-chain matmul that returns BOTH the
intermediate result `%1` and the final result `%2`. IREE's stream
scheduler packs the two values into a single transient buffer at offsets
0 and 128, so the second dispatch binds `%1` (input) and `%2` (output)
to the same root BO at different `byte_offset`s.

If the amdxdna HAL (`direct_command_buffer.cc::normal_run`) drops
either the buffer's subview `byte_offset` or the binding-level `offset`
when constructing firmware args, the two bindings collapse to the same
paddr, the second dispatch reads and writes the wrong slot, and one of
the two returned outputs ends up all-zero. With the fix, both outputs
match the expected matmul result on every run. Pre-fix this fails 100%
of runs (compile-time deterministic); post-fix it passes 100% of runs.

Asserts both outputs match the closed-form matmul results numerically
(via `run_and_verify`), so any future bug that produces wrong-but-non-
zero data is also caught.
"""
import pathlib

import numpy as np
import pytest

from conftest import IOSpec

THIS_DIR = pathlib.Path(__file__).resolve().parent
MLIR = THIS_DIR / "byte_offset_propagation.mlir"


@pytest.mark.regression
def test_byte_offset_propagation(
    compile_aie, run_and_verify, n_core_rows, n_core_cols, tmp_path
):
    # Inputs: 8x8 of 1.5s, 8x4 of 2.0s. Closed-form:
    #   %1 = lhs @ rhs  → 8x4, each element = 8 * 1.5 * 2.0  = 24.0
    #   %2 = lhs @ %1   → 8x4, each element = 8 * 1.5 * 24.0 = 288.0
    lhs = np.full((8, 8), 1.5, dtype=np.float32)
    rhs = np.full((8, 4), 2.0, dtype=np.float32)
    expected_1 = lhs @ rhs
    expected_2 = lhs @ expected_1

    vmfb = compile_aie(MLIR, tmp_path / "byte_offset_propagation.vmfb")
    spec = IOSpec(
        function_name="matmul_dual_out",
        inputs=[lhs, rhs],
        expected=[expected_1, expected_2],
    )
    run_and_verify(vmfb, spec, n_core_rows, n_core_cols, tmp_path)
