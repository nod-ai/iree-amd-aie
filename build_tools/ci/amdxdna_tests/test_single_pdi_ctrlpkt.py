# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""High-iteration soak test for the amdxdna ctrlpkt single-PDI dispatch path.

Compiles `matmul_f32_8_4_8` (the 4-chain matmul the original CI flakiness
was filed against) once, then runs iree-run-module `--n-iters` times in
a tight subprocess loop. Each iter is validated NUMERICALLY against the
closed-form matmul result, not just against non-zero.

This catches:

1. Per-binding `byte_offset` propagation regressions
   (direct_command_buffer.cc): when IREE's stream scheduler packs SSA
   values into a single transient BO, dropping the bindings'
   `byte_offset` makes inner dispatches read/write the wrong slot.
   Surfaces as wrong (sometimes non-zero) output, which the numerical
   check catches but a "non-zero" oracle would miss for this design.

2. iree-tooling's deferred output-readback path: relies on a transient
   `copy_buffer` whose recording lives inside an arena that the async
   queue worker may release back to the block pool before its
   post-release field clears. The next allocator to grab that block
   races with those trailing writes and the recorded
   `copy_buffer.header.type` can flip from COPY_BUFFER(7) to
   EXECUTION_BARRIER(0), silently dropping the copy and producing
   all-zero output. Surfaces at ~0.05% per-iter rate; 5000 iters gives
   >99% catch probability.
"""
import pathlib

import numpy as np
import pytest

from conftest import IOSpec

THIS_DIR = pathlib.Path(__file__).resolve().parent
# Reuse the same MLIR design the run.py MultipleDispatches suite uses.
MLIR_SRC = THIS_DIR.parent / "cpu_comparison" / "test_files" / "matmul_f32_8_4_8.mlir"


@pytest.mark.soak
def test_single_pdi_ctrlpkt(
    compile_aie, run_iree, n_core_rows, n_core_cols, n_iters, tmp_path
):
    if not MLIR_SRC.exists():
        pytest.skip(f"source MLIR not found at {MLIR_SRC}")

    # matmul_f32_8_4_8 is a 4-chain matmul iteratively applying lhs to rhs.
    # With lhs=1.5 (8x8) and rhs=2.0 (8x4):
    #   %1 each = 8 * 1.5 * 2.0    = 24.0
    #   %2 each = 8 * 1.5 * 24.0   = 288.0
    #   %3 each = 8 * 1.5 * 288.0  = 3456.0
    #   %4 each = 8 * 1.5 * 3456.0 = 41472.0
    lhs = np.full((8, 8), 1.5, dtype=np.float32)
    rhs = np.full((8, 4), 2.0, dtype=np.float32)
    expected = lhs @ rhs
    for _ in range(3):
        expected = lhs @ expected

    vmfb = compile_aie(MLIR_SRC, tmp_path / "matmul_f32_8_4_8.vmfb")
    spec = IOSpec(
        function_name="matmul_8_4_8",
        inputs=[lhs, rhs],
        expected=[expected],
    )

    failures = []
    for i in range(1, n_iters + 1):
        (got,) = run_iree(
            vmfb, spec, n_core_rows, n_core_cols, tmp_path, iter_tag=str(i)
        )
        if not np.allclose(got, expected, rtol=spec.rtol, atol=spec.atol):
            failures.append((i, got.flatten()[:4].tolist()))

    assert not failures, (
        f"{len(failures)}/{n_iters} soak iterations had numerical mismatch. "
        f"First few: {failures[:5]}"
    )
