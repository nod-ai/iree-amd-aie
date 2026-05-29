# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared pytest configuration + helpers for the amdxdna tests.

Tests in this directory are auto-discovered by pytest (any `test_*.py`).
Use markers to classify a test:

    @pytest.mark.regression  # deterministic regression test for a fixed bug
    @pytest.mark.soak        # high-iteration soak / stress / race detection

CI selects with `-m regression` or `-m soak`. Adding a new test file
with the right marker requires no CI workflow change.

The fixtures below give tests a small, opinionated API:

    def test_my_design(compile_aie, run_iree, ...):
        vmfb = compile_aie(MY_MLIR, tmp_path / "x.vmfb", CompileConfig(...))
        spec = IOSpec(function_name="...", inputs=[...], expected=[...])
        run_iree(vmfb, spec, n_core_rows, n_core_cols, tmp_path)

Common flag / sync / compare logic lives here once. Add a test → write
~10 lines.
"""
import dataclasses
import pathlib
import subprocess
import sys
import typing

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CLI options + path fixtures
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--iree-install-dir",
        required=True,
        type=pathlib.Path,
        help="Directory containing bin/iree-compile and bin/iree-run-module.",
    )
    parser.addoption(
        "--peano-dir",
        required=True,
        type=pathlib.Path,
        help="Directory containing the peano (llvm-aie) install.",
    )
    parser.addoption(
        "--target-device",
        default="npu4",
        help="iree-amdaie target device (npu1_4col, npu4, ...).",
    )
    parser.addoption(
        "--n-core-rows", type=int, default=4, help="amdxdna core-row count."
    )
    parser.addoption(
        "--n-core-cols", type=int, default=8, help="amdxdna core-col count."
    )
    parser.addoption(
        "--n-iters",
        type=int,
        default=5000,
        help="Default iteration count for `@pytest.mark.soak` tests "
        "(ignored by regression tests).",
    )
    parser.addoption(
        "--failure-log-dir",
        type=pathlib.Path,
        default=None,
        help="If set, persists iree-run-module stderr from failing iters to "
        "this directory. Intended for CI to upload as a workflow artifact.",
    )


@pytest.fixture
def iree_install_dir(request):
    return request.config.getoption("--iree-install-dir")


@pytest.fixture
def peano_dir(request):
    return request.config.getoption("--peano-dir")


@pytest.fixture
def target_device(request):
    return request.config.getoption("--target-device")


@pytest.fixture
def n_core_rows(request):
    return request.config.getoption("--n-core-rows")


@pytest.fixture
def n_core_cols(request):
    return request.config.getoption("--n-core-cols")


@pytest.fixture
def n_iters(request):
    return request.config.getoption("--n-iters")


@pytest.fixture
def failure_log_dir(request):
    d = request.config.getoption("--failure-log-dir")
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def iree_compile_bin(iree_install_dir):
    return iree_install_dir / "bin" / "iree-compile"


@pytest.fixture
def iree_run_bin(iree_install_dir):
    return iree_install_dir / "bin" / "iree-run-module"


# ---------------------------------------------------------------------------
# CompileConfig + compile_aie fixture
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CompileConfig:
    """Knobs for iree-compile invocations through the `compile_aie` fixture.

    Defaults match the original CI-failing ctrlpkt single-core design. Override
    fields per-test as needed; `extra_flags` appends raw `--iree-...=...`
    strings for anything not covered.
    """

    num_rows: int = 1
    num_cols: int = 1
    tile_pipeline: str = "pack-peel"
    lower_to_aie_pipeline: str = "objectFifo"
    enable_ctrlpkt: bool = True
    packet_flow_strategy: str = "auto"
    optimize_bindings: bool = False
    memoization: bool = False
    indirect_command_buffers: bool = False
    extra_flags: typing.Tuple[str, ...] = ()


@pytest.fixture
def compile_aie(
    iree_compile_bin,
    iree_install_dir,
    peano_dir,
    target_device,
    failure_log_dir,
    request,
):
    """Factory: `(mlir, out_vmfb, cfg=CompileConfig()) -> Path`.

    Centralizes the iree-compile flag list so a new flag added here
    propagates to every test using this fixture. On compile failure,
    persists stderr to `--failure-log-dir` (if set) for CI artifact
    upload.
    """

    def _compile(mlir, out_vmfb, cfg=None):
        if cfg is None:
            cfg = CompileConfig()
        cmd = [
            str(iree_compile_bin),
            str(mlir),
            "--iree-hal-target-backends=amd-aie",
            f"--iree-amdaie-target-device={target_device}",
            f"--iree-amdaie-tile-pipeline={cfg.tile_pipeline}",
            f"--iree-amdaie-lower-to-aie-pipeline={cfg.lower_to_aie_pipeline}",
            f"--iree-amd-aie-peano-install-dir={peano_dir}",
            f"--iree-amd-aie-install-dir={iree_install_dir}",
            f"--iree-amdaie-num-rows={cfg.num_rows}",
            f"--iree-amdaie-num-cols={cfg.num_cols}",
            "--iree-amdaie-device-hal=amdxdna",
            f"--iree-scheduling-optimize-bindings={str(cfg.optimize_bindings).lower()}",
            f"--iree-hal-memoization={str(cfg.memoization).lower()}",
            f"--iree-hal-indirect-command-buffers={str(cfg.indirect_command_buffers).lower()}",
        ]
        if cfg.enable_ctrlpkt:
            cmd += [
                "--iree-amdaie-enable-control-packet=true",
                f"--iree-amdaie-packet-flow-strategy={cfg.packet_flow_strategy}",
            ]
        cmd += list(cfg.extra_flags)
        cmd += ["-o", str(out_vmfb)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            _persist_failure_log(
                failure_log_dir, request.node.name, "compile", cmd, r.stderr
            )
            print(r.stderr, file=sys.stderr)
            pytest.fail(f"iree-compile exited rc={r.returncode}")
        return out_vmfb

    return _compile


# ---------------------------------------------------------------------------
# IOSpec + run_and_verify fixture
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IOSpec:
    """One iree-run-module invocation: function name, numpy inputs and
    expected outputs. Tests construct this once; `run_iree` and
    `assert_outputs_match` consume it.
    """

    function_name: str
    inputs: typing.List[np.ndarray]
    expected: typing.List[np.ndarray]
    rtol: float = 1e-6
    atol: float = 1e-6


def _arg_for(arr: np.ndarray) -> str:
    """iree-run-module `--input=` shape/dtype string for a numpy array."""
    shape = "x".join(str(d) for d in arr.shape)
    # iree-run-module spelling for the common dtypes the tests use.
    dtype_map = {
        np.dtype("float32"): "f32",
        np.dtype("int32"): "i32",
        np.dtype("int8"): "i8",
    }
    return f"{shape}x{dtype_map[arr.dtype]}"


@pytest.fixture
def run_iree(iree_run_bin, failure_log_dir, request):
    """Factory: `(vmfb, spec, n_core_rows, n_core_cols, tmp_path) ->
    List[np.ndarray]`.

    Writes spec.inputs to tmp_path bin files, invokes iree-run-module with
    matching --input/--output flags, reads the output bin files back as
    numpy arrays (shapes/dtypes from spec.expected), and returns them. On
    non-zero exit code or short output, fails the test and (if
    --failure-log-dir is set) persists stderr for CI artifact upload.
    """

    def _run(vmfb, spec: IOSpec, n_core_rows, n_core_cols, tmp_path, iter_tag=""):
        in_paths = []
        for i, arr in enumerate(spec.inputs):
            p = tmp_path / f"in{i}.bin"
            p.write_bytes(np.ascontiguousarray(arr).tobytes())
            in_paths.append(p)
        out_paths = [tmp_path / f"out{i}.bin" for i in range(len(spec.expected))]

        cmd = [str(iree_run_bin), f"--module={vmfb}"]
        for arr, p in zip(spec.inputs, in_paths):
            cmd.append(f"--input={_arg_for(arr)}=@{p}")
        cmd += [
            "--device=amdxdna",
            f"--amdxdna_n_core_rows={n_core_rows}",
            f"--amdxdna_n_core_cols={n_core_cols}",
            f"--function={spec.function_name}",
        ]
        for p in out_paths:
            cmd.append(f"--output=@{p}")

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            _persist_failure_log(
                failure_log_dir, request.node.name, iter_tag, cmd, r.stderr
            )
            print(r.stderr, file=sys.stderr)
            pytest.fail(
                f"iree-run-module exited rc={r.returncode}"
                + (f" (iter {iter_tag})" if iter_tag else "")
            )

        outputs = []
        for p, exp in zip(out_paths, spec.expected):
            raw = p.read_bytes()
            need = exp.size * exp.dtype.itemsize
            if len(raw) < need:
                _persist_failure_log(
                    failure_log_dir, request.node.name, iter_tag, cmd, r.stderr
                )
                pytest.fail(
                    f"output {p.name} short: {len(raw)}B < expected {need}B"
                    + (f" (iter {iter_tag})" if iter_tag else "")
                )
            outputs.append(
                np.frombuffer(raw[:need], dtype=exp.dtype).reshape(exp.shape)
            )
        return outputs

    return _run


@pytest.fixture
def run_and_verify(run_iree):
    """Factory: `(vmfb, spec, n_core_rows, n_core_cols, tmp_path) -> None`.

    Calls run_iree and asserts every output matches `spec.expected`
    elementwise within `spec.rtol/atol`. Use this for single-shot
    regression tests where one numerical mismatch should fail loudly.
    For soak loops where you want to count and report many iterations,
    use run_iree directly + numpy.allclose.
    """

    def _verify(vmfb, spec, n_core_rows, n_core_cols, tmp_path):
        got = run_iree(vmfb, spec, n_core_rows, n_core_cols, tmp_path)
        for i, (g, exp) in enumerate(zip(got, spec.expected)):
            np.testing.assert_allclose(
                g,
                exp,
                rtol=spec.rtol,
                atol=spec.atol,
                err_msg=(
                    f"output[{i}] mismatch for {spec.function_name}: "
                    f"got first 4: {g.flatten()[:4]}, "
                    f"expected: {exp.flatten()[:4]}"
                ),
            )

    return _verify


def _persist_failure_log(failure_log_dir, test_name, iter_tag, cmd, stderr_text):
    if failure_log_dir is None:
        return
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in test_name)
    suffix = f"_iter{iter_tag}" if iter_tag else ""
    path = failure_log_dir / f"{safe}{suffix}.log"
    with path.open("w") as f:
        f.write("COMMAND:\n" + " ".join(cmd) + "\n\nSTDERR:\n" + (stderr_text or ""))
