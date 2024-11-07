#!/usr/bin/env python3

# Copyright 2024 The IREE Authors

import argparse
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent
import numpy as np
from convolution_template.convolution_generator import ConvolutionMlirGenerator
from matmul_template.matmul_generator import generate_matmul_test
from output_comparer import compare
from input_generator import (
    generate_inputs,
    verify_determinism,
    load_input,
    get_output_type,
    np_from_binfile,
    f32_to_bf16,
    bf16_to_f32,
)


class ConvolutionFromTemplate:
    def __init__(self, params):
        self.generator = ConvolutionMlirGenerator(**params)
        params = self.generator.params
        conv_type = params["conv_type"]
        N = params["N"]
        IW = params["IW"]
        in_type = params["input_element_type"]
        out_type = params["output_element_type"]
        # TODO(newling) Use all parameters in name, to avoid name collision.
        self.name = f"{conv_type}_{N}_{IW}_{in_type}_{out_type}"
        self.labels = ["Convolution"]

    def run(self, config):
        # Generate MLIR file:
        output_dir = config.output_dir
        filename = output_dir / f"{self.name}.mlir"
        self.generator.write_to_file(filename)
        aie_vs_llvm_cpu(
            config,
            filename,
            tile_pipeline="conv-decompose",
            lower_to_aie_pipeline="objectFifo",
            n_repeats=2,
        )
        # Return True to indicate that the test ran.
        return True


class ConvolutionNHWCQ:
    def __init__(self):
        self.name = "convolution_nhwc_q"
        self.labels = ["Convolution", "ConvolutionNHWCQ"]

    def run(self, config):
        files_dir = config.file_dir / "test_files"
        filename = files_dir / "conv2d_nhwc_q.mlir"
        aie_vs_llvm_cpu(
            config,
            filename,
            tile_pipeline="conv-decompose",
            lower_to_aie_pipeline="objectFifo",
            n_repeats=1,
        )
        return True


class MultipleDispatches:
    def __init__(self, name):
        self.name = name
        self.labels = ["Matmul", "MultipleDispatches"]

    def run(self, config):
        test_files_dir = config.file_dir / "test_files"
        self.filename = test_files_dir / f"{self.name}.mlir"
        if config.xdna_datetime and config.xdna_datetime < 20240801:
            aie_vs_llvm_cpu(config, self.filename, function_name="three_$mm$")
            return True
        else:
            # Return False to indicate that the test did not run.
            return False


class BaseMatmul:
    def __init__(self, M, N, K, input_type, acc_type):
        self.labels = []
        self.M = M
        self.N = N
        self.K = K
        self.input_type = input_type
        self.acc_type = acc_type
        self.labels.append("Matmul")


class MatmulFullBias(BaseMatmul):
    """
    A test of the form matmul(A,B) + C where A:MxK, B:KxN, C:MxN
    """

    def __init__(self, M, N, K, input_type, acc_type):
        super().__init__(M, N, K, input_type, acc_type)
        self.name = f"matmul_full_bias_{M}_{N}_{K}_{input_type}_{acc_type}"
        self.labels.append("MatmulFullBias")

    def run(self, config):
        filename = config.output_dir / f"{self.name}.mlir"
        matmul_template_dir = config.file_dir / "matmul_template"
        template_name = matmul_template_dir / "matmul_bias_MxK_KxN_MxN.mlir"
        generate_matmul_test(
            filename,
            template_name,
            self.M,
            self.N,
            self.K,
            self.input_type,
            self.acc_type,
        )
        aie_vs_llvm_cpu(
            config,
            filename,
            tile_pipeline="pack-peel",
            # TODO(someone) This should work for "objectFifo".
            lower_to_aie_pipeline="air",
        )

        return True


class VanillaMatmul(BaseMatmul):
    """
    A test of the form matmul(A,B) where A:MxK, B:KxN
    """

    def __init__(self, M, N, K, input_type, acc_type):
        super().__init__(M, N, K, input_type, acc_type)
        self.name = f"vanilla_matmul_{M}_{N}_{K}_{input_type}_{acc_type}"
        self.labels.append("VanillaMatmul")

    def run(self, config):
        self.filename = config.output_dir / f"{self.name}.mlir"
        matmul_template_dir = config.file_dir / "matmul_template"
        template_name = matmul_template_dir / "matmul_MxK_KxN.mlir"
        generate_matmul_test(
            self.filename,
            template_name,
            self.M,
            self.N,
            self.K,
            self.input_type,
            self.acc_type,
        )

        aie_vs_llvm_cpu(
            config,
            self.filename,
        )

        return True


class LegacyMatmul(BaseMatmul):
    """
    Ported from run_matmul_test.sh (Moving all tests over is work in progress).

    These tests use numpy as a baseline.
    """

    # TODO(newling) Make 'MxKxN' the default everywhere.
    def __init__(
        self,
        M,
        K,
        N,
        input_type,
        acc_type,
        n_runs=1,
        approximation_threshold=20000,
        seed=1,
        additional_labels=[],
    ):
        super().__init__(M, N, K, input_type, acc_type)
        self.n_runs = n_runs
        self.approximation_threshold = approximation_threshold
        self.seed = seed
        self.name = f"legacy_matmul_{M}_{N}_{K}_{input_type}_{acc_type}"
        self.labels.append("LegacyMatmul")
        self.labels += additional_labels

    def run_local(self, config, input_args, post_processor, baseline, raise_failure):
        return aie_vs_baseline(
            config,
            self.filename,
            input_args,
            baseline,
            use_ukernel=False,
            tile_pipeline="pack-peel",
            lower_to_aie_pipeline="objectFifo",
            function_name=None,
            seed=self.seed,
            rtol=0,
            atol=0,
            n_repeats=self.n_runs,
            output_type=get_output_type(self.filename),
            post_processor=post_processor,
            raise_failure=raise_failure,
        )

    def run(self, config):
        self.filename = config.output_dir / f"{self.name}.mlir"
        template_dir = config.file_dir / "matmul_template"
        template_name = template_dir / "matmul_MxK_KxN.mlir"
        generate_matmul_test(
            self.filename,
            template_name,
            self.M,
            self.N,
            self.K,
            self.input_type,
            self.acc_type,
        )

        input_args = generate_inputs(self.filename, config.output_dir, self.seed, {})

        a = load_input(input_args[0])
        b = load_input(input_args[1])

        M = self.M
        N = self.N
        K = self.K

        # To ensure we don't spend too long running matmul in numpy (for example
        # on my computer with M=N=K=1e4, it takes about 10 seconds to run), we
        # only do an approximate check that the output from AIE is correct.
        # This mimics the behaviour of the original run_matmul_test.sh script.
        #
        # The value self.approximation_threshold specifies a threshold, below
        # which we must run the full test.
        #
        # Suppose that self.approximation_threshold = 20000. Then if
        # 1) M*N < 20000, we run the full test.
        # 2) M*N <= 10 * 20000, we run the full test too (because
        #    weguess that sparsity only helps if it is < 0.1).
        # 3) M*N > 10 * 20000, we run an approximate test, sampling
        #    20000 elements from the output and comparing them to numpy
        #    computed inner products.
        approximate_success = False
        do_approximate = M * N > 10 * self.approximation_threshold
        if do_approximate:

            # Select the indices to check for correctness as random (without replacement).
            generator = get_generator(self.seed)
            indices = generator.choice(
                M * K, self.approximation_threshold, replace=False
            )
            indices.sort()

            # Compute the correct values for the selected indices.
            baseline_values = np.array(
                [np.dot(a[i // K], b[:, i % N]) for i in indices]
            )

            # Describe how to subsample the AIE output to compare with the baseline.
            post_processor = lambda X: np.array([X[i // N, i % N] for i in indices])

            # Run the approximate test. Do not raise an exception if the comparison
            # fails, because we'll run the full test in that case, so that we can
            # provide better diagnostics.
            raise_failure = False
            approximate_success = self.run_local(
                config, input_args, post_processor, baseline_values, raise_failure
            )

        # The approximate test either didn't run (because the test is small) or
        # it did run and it failed. In these cases, we run the full test.
        if not approximate_success:
            self.run_local(config, input_args, None, np.dot(a, b), config.raise_failure)

        # Return true because the test ran.
        return True


class MatmulThinBias(BaseMatmul):
    """
    A test of the form matmul(A,B) + C where A:MxK, B:KxN, C:N
    """

    def __init__(self, M, N, K, input_type, acc_type, use_ukernel):
        super().__init__(M, N, K, input_type, acc_type)
        tail = "" if use_ukernel else "ukernel"
        self.name = f"matmul_thin_bias_{M}_{N}_{K}_{input_type}_{acc_type}_{tail}"
        self.labels.append("MatmulThinBias")
        if use_ukernel:
            self.labels.append("UKernel")
        self.use_ukernel = use_ukernel

    def run(self, config):

        self.filename = config.output_dir / f"{self.name}.mlir"
        matmul_template_dir = config.file_dir / "matmul_template"
        template_name = matmul_template_dir / "matmul_bias_MxK_KxN_N.mlir"
        generate_matmul_test(
            self.filename,
            template_name,
            self.M,
            self.K,
            self.N,
            self.input_type,
            self.acc_type,
        )

        if self.use_ukernel and not config.vitis_dir:
            return False

        else:
            aie_vs_llvm_cpu(
                config,
                self.filename,
                tile_pipeline="pack-peel",
                # TODO(someone) This should work for "objectFifo".
                lower_to_aie_pipeline="air",
                use_ukernel=self.use_ukernel,
            )
            return True


class BatchMatmul(BaseMatmul):
    """
    A test of the form batch_matmul(A,B) where A:BxMxK, B:BxKxN
    """

    def __init__(self, B, M, N, K, input_type, acc_type):
        super().__init__(M, N, K, input_type, acc_type)

        self.name = f"batch_matmul_{B}_{M}_{N}_{K}_{input_type}_{acc_type}"
        self.labels.append("BatchMatmul")
        self.B = B

    def run(self, config):
        self.filename = config.output_dir / f"{self.name}.mlir"
        matmul_template_dir = config.file_dir / "matmul_template"
        template_name = matmul_template_dir / "batch_matmul_BxMxK_BxKxN.mlir"
        generate_matmul_test(
            self.filename,
            template_name,
            k=self.K,
            b=self.B,
            m=self.M,
            n=self.N,
            lhs_rhs_type=self.input_type,
            acc_type=self.acc_type,
        )
        aie_vs_llvm_cpu(
            config,
            self.filename,
        )

        return True


class MatmulTruncf(BaseMatmul):
    """
    A test of the form matmul(A,B) + truncf(C) where A:MxK, B:KxM and C:MxM
    """

    def __init__(self, M, K, input_type, acc_type, lhs, rhs, expected_out):
        super().__init__(M, M, K, input_type, acc_type)
        self.name = f"matmul_truncf_{M}_{K}_{input_type}_{acc_type}"
        self.labels.append("MatmulTruncf")
        self.lhs = lhs
        self.rhs = rhs
        self.expected_out = expected_out

        # Assertions on shapes: Check that lhs is MxK, rhs is KxM, and expected_out is MxM
        assert lhs.shape == (M, K)
        assert rhs.shape == (K, M)
        assert expected_out.shape == (M, M)

    def run(self, config):

        self.filename = config.output_dir / f"{self.name}.mlir"
        matmul_template_dir = config.file_dir / "matmul_template"
        template_name = matmul_template_dir / "matmul_truncf_MxK_KxN.mlir"
        generate_matmul_test(
            self.filename,
            template_name,
            self.M,
            self.N,
            self.K,
            self.input_type,
            self.acc_type,
        )

        input_args = generate_inputs(
            self.filename, config.output_dir, 1, {1: self.lhs, 2: self.rhs}
        )
        aie_vs_baseline(
            config,
            self.filename,
            input_args,
            self.expected_out,
            use_ukernel=False,
            tile_pipeline="pack-peel",
            lower_to_aie_pipeline="objectFifo",
            function_name=None,
            seed=1,
            rtol=0,
            atol=0,
            n_repeats=1,
            output_type=get_output_type(self.filename),
            post_processor=None,
            raise_failure=config.raise_failure,
        )

        return True


def find_executable(install_dir: Path, executable_name):
    """
    Search for an executable in the given directory and its subdirectories
    'bin' and 'tools'. If the executable is not found, raise a RuntimeError.
    """
    search_dirs = [
        install_dir,
        install_dir / "bin",
        install_dir / "tools",
    ]

    if platform.system() == "Windows":
        executable_name += ".exe"

    for directory in search_dirs:
        executable_path = directory / executable_name
        if executable_path.is_file():
            return executable_path
    raise RuntimeError(
        f"No '{executable_name}' executable found in '{install_dir}' or subdirectories."
    )


def shell_out(cmd: list, workdir=None, verbose: int = 0, raise_on_error=True, env=None):
    if workdir is None:
        workdir = Path.cwd()
    workdir = Path(workdir)
    os.chdir(workdir)
    if not isinstance(cmd, list):
        cmd = [cmd]
    for i, c in enumerate(cmd):
        if isinstance(c, Path):
            cmd[i] = str(c)
    if env is None:
        env = {}

    env = {**env, **os.environ}

    if verbose:
        _cmd = " ".join(cmd)
        if verbose > 1:
            _cmd = " ".join([f"{k}={v}" for k, v in env.items()]) + " " + _cmd
        print(f"Running the following command:\n{_cmd}")

    handle = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    stdout, stderr = handle.communicate()
    stderr_decode = stderr.decode("utf-8").strip()
    stdout_decode = stdout.decode("utf-8").strip()
    if verbose:
        if stdout_decode:
            print("Standard output from script:")
            print(stdout_decode)
        if stderr_decode:
            print("Standard error from script:", file=sys.stderr)
            print(stderr_decode, file=sys.stderr)
    if not raise_on_error and handle.returncode != 0:
        print(
            f"Error executing script, error code was {handle.returncode}. Not raising an error.",
            file=sys.stderr,
        )
    if raise_on_error and handle.returncode != 0:
        raise RuntimeError(
            f"Error executing script, error code was {handle.returncode}",
        )
    return stdout_decode, stderr_decode


def generate_aie_vmfb(
    config,
    name,
    tile_pipeline,
    lower_to_aie_pipeline,
    use_ukernel,
    test_file,
    input_args,
    function_name,
):
    """
    Compile a test file for IREE's AIE backend, returning the path to the
    compiled module.
    """

    additional_flags = config.additional_aie_compilation_flags.split()

    compilation_flags = [
        config.iree_compile_exe,
        test_file,
        "--iree-hal-target-backends=amd-aie",
        f"--iree-amdaie-target-device={config.target_device}",
        f"--iree-amdaie-tile-pipeline={tile_pipeline}",
        f"--iree-amdaie-lower-to-aie-pipeline={lower_to_aie_pipeline}",
        "--iree-amdaie-matmul-elementwise-fusion",
        f"--iree-amd-aie-peano-install-dir={config.peano_dir}",
        f"--iree-amd-aie-install-dir={config.iree_install_dir}",
        f"--iree-amd-aie-vitis-install-dir={config.vitis_dir}",
        f"--iree-amd-aie-enable-chess={int(config.use_chess)}",
        f"--iree-hal-dump-executable-files-to={config.output_dir}",
        f"--iree-amdaie-device-hal={config.device_hal}",
        "--iree-scheduling-optimize-bindings=false",
        "--iree-hal-memoization=false",
        "--iree-hal-indirect-command-buffers=false",
        f"--mlir-disable-threading",
        "--mlir-elide-resource-strings-if-larger=10",
    ]

    if config.verbose:
        compilation_flags += ["--iree-amd-aie-show-invoked-commands"]

    if use_ukernel:
        compilation_flags += ["--iree-amdaie-enable-ukernels=all"]

    for additional_flag in additional_flags:
        if additional_flag not in compilation_flags:
            compilation_flags += [additional_flag]

    compilation_flags += [
        "-o",
        config.output_dir / f"{name}_aie.vmfb",
    ]

    start = time.monotonic_ns()
    shell_out(compilation_flags, config.output_dir, config.verbose)
    compile_time = time.monotonic_ns() - start
    if config.verbose:
        print(f"Time spent in compilation: {compile_time // 1e6} [ms]")

    aie_vmfb = config.output_dir / f"{name}_aie.vmfb"
    if not aie_vmfb.exists():
        raise RuntimeError(f"Failed to compile {test_file} to {aie_vmfb}")

    return aie_vmfb


def generate_aie_output(config, aie_vmfb, input_args, function_name, name, output_type):
    """
    Run a compiled AIE module (aie_vmfb), returning a numpy array of the output.
    """

    aie_bin = config.output_dir / f"{name}_aie.bin"
    run_args = [
        config.iree_run_exe,
        f"--module={aie_vmfb}",
        *input_args,
        f"--device={config.device_hal}",
        f"--output=@{aie_bin}",
    ]
    if function_name:
        run_args += [f"--function={function_name}"]
    if config.xrt_lite_n_core_rows is not None:
        run_args += [f"--xrt_lite_n_core_rows={config.xrt_lite_n_core_rows}"]
    if config.xrt_lite_n_core_cols is not None:
        run_args += [f"--xrt_lite_n_core_cols={config.xrt_lite_n_core_cols}"]

    if config.reset_npu_between_runs:
        shell_out(config.reset_npu_script, verbose=config.verbose)

    start = time.monotonic_ns()
    shell_out(run_args, config.output_dir, config.verbose)
    run_time = time.monotonic_ns() - start

    if config.verbose:
        print(f"Time spent in running the model: {run_time // 1e6} [ms]")

    return np_from_binfile(aie_bin, output_type)


def generate_llvm_cpu_output(
    config,
    name,
    test_file,
    input_args,
    function_name,
    output_type,
):
    """
    Compile and run a test file for IREE's CPU backend, returning a numpy array
    of the output.
    """

    cpu_vmfb = config.output_dir / f"{name}_cpu.vmfb"
    compilation_flags = [
        config.iree_compile_exe,
        test_file,
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu-features=host",
        "-o",
        f"{cpu_vmfb}",
    ]
    shell_out(compilation_flags, workdir=config.output_dir, verbose=config.verbose)

    cpu_bin = config.output_dir / f"{name}_cpu.bin"
    run_args = [
        config.iree_run_exe,
        f"--module={cpu_vmfb}",
        *input_args,
        f"--output=@{cpu_bin}",
    ]
    if function_name:
        run_args += [f"--function={function_name}"]
    shell_out(run_args, workdir=config.output_dir, verbose=config.verbose)
    return np_from_binfile(cpu_bin, output_type)


class TestConfig:
    """
    Global state used for all tests. Stores paths to executables used, and
    records test failures.
    """

    def __init__(
        self,
        output_dir,
        iree_install_dir,
        peano_dir,
        xrt_dir,
        vitis_dir,
        file_dir,
        iree_compile_exe,
        iree_run_exe,
        verbose,
        raise_failure,
        reset_npu_between_runs,
        do_not_run_aie,
        additional_aie_compilation_flags,
        device_hal,
        xrt_lite_n_core_rows,
        xrt_lite_n_core_cols,
        target_device,
        use_chess,
    ):
        self.output_dir = output_dir
        self.iree_install_dir = iree_install_dir
        self.peano_dir = peano_dir
        self.xrt_dir = xrt_dir
        self.vitis_dir = vitis_dir
        self.file_dir = file_dir
        self.iree_compile_exe = iree_compile_exe
        self.iree_run_exe = iree_run_exe
        self.raise_failure = raise_failure
        self.verbose = verbose
        self.xdna_datetime = None
        self.xdna_hash = None
        self.reset_npu_between_runs = reset_npu_between_runs
        self.do_not_run_aie = do_not_run_aie
        self.additional_aie_compilation_flags = additional_aie_compilation_flags
        self.device_hal = device_hal
        self.xrt_lite_n_core_rows = xrt_lite_n_core_rows
        self.xrt_lite_n_core_cols = xrt_lite_n_core_cols
        self.target_device = target_device
        self.use_chess = use_chess

        # Try get the xrt and (linux) kernel versions.
        self.linux_kernel = "undetermined"
        self.xrt_hash_date = "undetermined"
        self.xrt_hash = "undetermined"
        self.xdna_hash = "undetermined"
        self.xrt_release = "undetermined"
        self.peano_commit_hash = "undetermined"

        self.reset_npu_script = file_dir.parent / "reset_npu.sh"
        if reset_npu_between_runs and not self.reset_npu_script.exists():
            raise RuntimeError(
                f"The file {self.reset_npu_script} does not exist, and reset_npu_script=True"
            )

        # Populated at runtime
        self.failures = []

        if not isinstance(self.verbose, bool) and not isinstance(self.verbose, int):
            raise ValueError(
                f"verbose must be a boolean or integer, not {type(verbose)}"
            )

        if not xrt_dir:
            return

        xrt_bin_dir = xrt_dir / "bin"
        xrt_smi_exe = xrt_bin_dir / "xrt-smi"
        if not xrt_smi_exe.exists():
            xrt_smi_exe = xrt_bin_dir / "xbutil"
        if not xrt_smi_exe.exists():
            raise RuntimeError(f"Neither xrt-smi nor xbutil found in {xrt_bin_dir}")

        # Get the string output of the xrt-smi 'examine' command. Expect the
        # string to look something like:
        #
        # ```
        # System Configuration
        # OS Name              : Linux
        # Release              : 6.9.1-20240521t190425.46e42a4
        # ...
        #
        # XRT
        # Version              : 2.18.71
        # Hash Date            : 2024-07-08 20:13:41
        # ...
        # ```
        #
        system_info, xrt_info = (
            subprocess.check_output([xrt_smi_exe, "examine"])
            .decode("utf-8")
            .split("XRT")
        )

        linux_kernel = re.findall(r"Release\s+:\s(.*)", system_info, re.MULTILINE)
        if linux_kernel:
            self.linux_kernel = linux_kernel[0]

        xrt_release = re.findall(r"Version\s+:\s(.*)", xrt_info, re.MULTILINE)
        if xrt_release:
            self.xrt_release = xrt_release[0]

        xrt_hash_date = re.findall(r"Hash Date\s+:\s(.*)", xrt_info, re.MULTILINE)
        if xrt_hash_date:
            self.xrt_hash_date = xrt_hash_date[0]

        xrt_hash = re.findall(r"Hash\s+:\s(.*)", xrt_info, re.MULTILINE)
        if xrt_hash:
            self.xrt_hash = xrt_hash[0]

        xdna_datetime_hash = re.findall(
            # eg 2.18.0_20240606
            r"amdxdna\s+:\s\d\.\d+\.\d+_(\d+),\s(\w+)",
            xrt_info,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        if xdna_datetime_hash:
            self.xdna_datetime = int(xdna_datetime_hash[0][0])
            self.xdna_hash = xdna_datetime_hash[0][1]

        # Try and get the peano commit hash. This is a bit of a hack, if it fails
        # peano_commit_has is left as "undetermined".
        peano_clang_path = peano_dir / "bin" / "clang"
        if peano_clang_path.exists():
            _, clang_v_output = shell_out(
                [peano_clang_path, "-v"], verbose=self.verbose, raise_on_error=False
            )
            peano_commit_hash = re.findall(
                r"clang version \d+\.\d+\.\d+ \(https://github.com/Xilinx/llvm-aie (\w+)\)",
                clang_v_output,
                re.MULTILINE,
            )
            if peano_commit_hash:
                self.peano_commit_hash = peano_commit_hash[0]

    def __str__(self):
        return dedent(
            f"""
        Settings and versions used in all tests
        -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
        target_device:        {self.target_device}
        do_not_run_aie:       {self.do_not_run_aie}
        file_dir:             {self.file_dir}
        iree_compile_exe:     {self.iree_compile_exe}
        iree_install_dir:     {self.iree_install_dir}
        iree_run_exe:         {self.iree_run_exe}
        kernel_version:       {self.linux_kernel}
        output_dir:           {self.output_dir}
        peano_commit_hash:    {self.peano_commit_hash}
        peano_dir:            {self.peano_dir}
        reset_npu_script:     {self.reset_npu_script}
        raise_failure:       {self.raise_failure}
        use_chess:            {self.use_chess}
        verbose:              {self.verbose}
        vitis_dir:            {self.vitis_dir}
        xrt_dir:              {self.xrt_dir}
        xrt_hash_date:        {self.xrt_hash_date}
        xrt_hash:             {self.xrt_hash}
        xrt_release:          {self.xrt_release}
        xdna_hash_date:       {self.xdna_datetime}
        xdna_hash:            {self.xdna_hash}
        -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

        Some information on the above settings / versions
        =================================================
        target_device
          The NPU device to be targeted (npu1_4col, npu4).
        do_not_run_aie
          If True, then the AIE backend will not be run. This is useful for
          ensuring that everything up to the AIE run and numerical comparison
          is working correctly, for example if you are not on a device with
          working AIE HW and runtime.
        file_dir
          The directory of this script
        iree_compile_exe
          The path to the IREE compiler executable used to compile for both
          AIE and CPU backends
        kernel_version
          The version of the linux kernel. We try to get this by running
          'xrt_smi examine' or 'xbutil examine'
        peano_commit_hash
          The version of peano (llvm-aie). We try to get this by looking
          for a 'dist-info' directory which carries a wheel's date. This is
          just a hint, and it's not guaranteed to be correct. This is not
          used in the tests, but included here as it might be useful for debugging.
        xrt_hash/date
          Information obtained from 'xrt_smi examine' or 'xbutil examine' about
          the version of XRT. This is not used in the tests, but included here
          just to help in debugging.
        =================================================
        """
        )

    def __repr__(self):
        return self.__str__()


def name_from_mlir_filename(mlir_filename):
    return os.path.basename(mlir_filename).replace(".mlir", "")


def aie_vs_baseline(
    config,
    test_file,
    input_args,
    baseline_value,
    use_ukernel,
    tile_pipeline,
    lower_to_aie_pipeline,
    function_name,
    seed,
    rtol,
    atol,
    n_repeats,
    output_type,
    post_processor,
    raise_failure,
):
    """
    If the outputs differ, add the test file to a list of failures.

    Arguments to the function are:
    config:
        TestConfig containing any state which is common to all tests
    test_file:
        The path to the test (.mlir) file
    input_args:
        a string of the form
        "--input=3x40xf32=@<binary_file> --input=2x2xi32=@<binary_file>"
    baseline_value:
        The expected output of running the test file through the AIE
        backend. Computed any which way you like
    use_ukernel:
        Whether to use micro-kernels when running on the AIE backend
    tile_pipeline:
        The tiling pipeline to use when compiling for the AIE backend
    function_name:
        The name of the function to run (the test file may contain multiple
        functions)
    ...
    n_repeats:
        The number of times to run the test. This is useful for tests which
        may pass only sometimes due to driver issues, etc.
    ...
    post_processor:
       filter down to final values.
    """

    name = name_from_mlir_filename(test_file)

    aie_vmfb = generate_aie_vmfb(
        config,
        name,
        tile_pipeline,
        lower_to_aie_pipeline,
        use_ukernel,
        test_file,
        input_args,
        function_name,
    )

    if config.do_not_run_aie:
        if config.verbose:
            print(f"Skipping AIE run for {test_file} because 'do_not_run_aie=True'.")
        return

    for i in range(n_repeats):
        if config.verbose:
            print(f"Run #{i + 1} of {n_repeats} for {test_file}")

        aie_output = generate_aie_output(
            config,
            aie_vmfb,
            input_args,
            function_name,
            name,
            output_type,
        )

        final_aie_output = aie_output
        if post_processor:
            final_aie_output = post_processor(aie_output)

        summary_string = compare(baseline_value, final_aie_output, rtol, atol)
        if summary_string:
            print(summary_string)
            config.failures.append(test_file)
            if raise_failure:
                raise RuntimeError("Test failed, exiting.")
            else:
                return False

    # True indicates test passed
    return True


def aie_vs_llvm_cpu(
    config,
    test_file,
    use_ukernel=False,
    tile_pipeline="pack-peel",
    lower_to_aie_pipeline="objectFifo",
    function_name=None,
    seed=1,
    rtol=1e-6,
    atol=1e-6,
    n_repeats=1,
):
    """
    Compare the output obtained when compiling and running on IREE's
    (nascent) AIE and (more mature) llvm-cpu backends.
    """

    if n_repeats == 0:
        return

    name = name_from_mlir_filename(test_file)
    if config.verbose:
        print(f"Running {name} test")

    input_args = generate_inputs(test_file, config.output_dir, seed)
    output_type = get_output_type(test_file)

    cpu_output = generate_llvm_cpu_output(
        config,
        name,
        test_file,
        input_args,
        function_name,
        output_type,
    )

    aie_vs_baseline(
        config,
        test_file,
        input_args,
        cpu_output,
        use_ukernel,
        tile_pipeline,
        lower_to_aie_pipeline,
        function_name,
        seed,
        rtol,
        atol,
        n_repeats,
        output_type,
        post_processor=None,
        raise_failure=config.raise_failure,
    )


def nearest_bf16_below(x):
    vals = np.array([x], dtype=np.float32)
    return bf16_to_f32(f32_to_bf16(vals))[0]


def get_matmul_truncf_tests():

    M = 128
    K = 256
    a = 2
    b = 3
    test1 = MatmulTruncf(
        M,
        K,
        "bf16",
        "f32",
        a * np.ones([M, K]),
        b * np.ones([K, M]),
        nearest_bf16_below(a * b * K) * np.ones([M, M]),
    )

    M = 8
    K = None
    a = 101
    b = 3
    test0 = MatmulTruncf(
        M,
        M,
        "bf16",
        "f32",
        a * np.ones([M, M]),
        b * np.eye(M),
        nearest_bf16_below(a * b) * np.ones([M, M]),
    )

    return [test0, test1]


class Tests:

    def register(self, test):
        self.tests.append(test)
        if test.name in self.existing_names:
            raise ValueError(f"Test name {test.name} is not unique")
        self.existing_names.append(test.name)

    def get_label_set(self):
        """
        Get the set of all the labels that are used in the tests.
        """
        labels = set()
        for test in self.tests:
            for label in test.labels:
                labels.add(label)
        labels = list(labels)
        labels.sort()
        return labels

    def get_test_names(self):
        names = [test.name for test in self.tests]
        names.sort()
        return names

    def __init__(self):
        self.existing_names = []
        self.tests = []

        # Matmul with truncf test(s):
        for test in get_matmul_truncf_tests():
            self.register(test)

        # BatchMatmul test(s):
        for input_type, acc_type in zip(["i32", "bf16"], ["i32", "f32"]):
            # Batch size = 1:
            self.register(BatchMatmul(1, 128, 128, 256, input_type, acc_type))
            # Batch size = 2:
            self.register(BatchMatmul(2, 64, 64, 64, input_type, acc_type))

        # MatmulThinBias test(s):
        self.register(MatmulThinBias(1024, 1024, 512, "bf16", "f32", True))
        self.register(MatmulThinBias(1024, 1024, 512, "bf16", "f32", False))

        # VanillaMatmul test(s):
        self.register(VanillaMatmul(32, 32, 32, "i32", "i32"))
        self.register(VanillaMatmul(32, 32, 64, "bf16", "f32"))

        # LegacyMatmul test(s):
        # Run repeatedly to check for non-deterministic hangs and numerical
        # issues.
        self.register(LegacyMatmul(32, 32, 32, "i32", "i32", n_runs=1000))

        i32_shapes_small = [
            "32x32x32",
            "64x32x128",
            "128x32x64",
            "128x32x64",
            "128x32x128",
            "256x32x256",
            "32x64x32",
            "64x64x64",
            "128x256x128",
        ]
        for s in i32_shapes_small:
            [M, K, N] = [int(x) for x in s.split("x")]
            self.register(
                LegacyMatmul(
                    M,
                    K,
                    N,
                    "i32",
                    "i32",
                    n_runs=10,
                    additional_labels=["small_matmul_i32"],
                )
            )

        # MultipleDispatches tests:
        for name in ["two_matmul_switching", "matmul_f32_8_8_4", "matmul_f32_8_4_8"]:
            self.register(MultipleDispatches(name))

        # MatmulFullBias test:
        self.register(MatmulFullBias(128, 128, 256, "i32", "i32"))

        # Convolution NHCWQ test:
        self.register(ConvolutionNHWCQ())

        # Convolution 2D tests:
        conv_2d_map = {
            "conv_type": "conv_2d_nhwc_hwcf",
            "N": 2,
            "IH": 14,
            "IC": 32,
            "OC": 64,
            "KH": 3,
        }
        for input_type, output_type in zip(
            ["i32", "bf16", "i8"], ["i32", "f32", "i32"]
        ):
            conv_2d_map["input_element_type"] = input_type
            conv_2d_map["output_element_type"] = output_type
            self.register(ConvolutionFromTemplate(conv_2d_map))

        # Depthwise convolution tests:
        depthwise_map = {
            "conv_type": "depthwise_conv_2d_nhwc_hwc",
            "N": 1,
            "IH": 14,
            "IC": 64,
            "KH": 3,
            "input_element_type": "i32",
            "output_element_type": "i32",
        }
        self.register(ConvolutionFromTemplate(depthwise_map))


def all_tests(
    tests,
    output_dir,
    iree_install_dir,
    peano_dir,
    xrt_dir,
    vitis_dir,
    raise_failure,
    verbose,
    reset_npu_between_runs,
    do_not_run_aie,
    test_set,
    additional_aie_compilation_flags,
    device_hal,
    xrt_lite_n_core_rows,
    xrt_lite_n_core_cols,
    target_device,
    use_chess,
):
    """
    There are a few ways to add tests to this script:

    1) add a single test file in `./test_files` which should follow the same
       format as the example `./test_files/matmul_int32.mlir`.

    2) use an existing template in `./matmul_template` to generate a test file
       with a fixed structure. Currently a handful of matmul templates exist in
       that directory.

    3) create a new matmul template in `./matmul_template`, for example if you
       want to add a new variant with tranposed operands or unary elementwise
       operations.

    4) create a new template generator, duplicating the directory structure of
       ./matmul_template. For example you might want to create ./conv_template
    """

    if not output_dir.exists():
        output_dir.mkdir()
    if not iree_install_dir.exists():
        raise RuntimeError(f"'{iree_install_dir}' is not a directory.")
    iree_compile_exe = find_executable(iree_install_dir, "iree-compile")
    iree_run_exe = find_executable(iree_install_dir, "iree-run-module")
    file_dir = Path(__file__).parent

    config = TestConfig(
        output_dir,
        iree_install_dir,
        peano_dir,
        xrt_dir,
        vitis_dir,
        file_dir,
        iree_compile_exe,
        iree_run_exe,
        verbose,
        raise_failure,
        reset_npu_between_runs,
        do_not_run_aie,
        additional_aie_compilation_flags,
        device_hal,
        xrt_lite_n_core_rows,
        xrt_lite_n_core_cols,
        target_device,
        use_chess,
    )
    if verbose:
        print(config)

    # Sanity check that results are reproducible across platforms
    verify_determinism()

    # Verify a very basic script runs before running the more complex tests
    if platform.system() != "Windows":
        shell_out(["pwd"], verbose=config.verbose)

    # For each test in test_set, find the partition it belongs to and run it
    # if no partition is found, raise error.

    match_run = []
    match_not_run = []
    not_match = []

    for test in tests.tests:

        # Determine if the test is a match for the test_set provided by caller
        match = "All" in test_set
        match = match or test.name in test_set
        for label in test.labels:
            match = match or label in test_set

        if match:
            did_run = test.run(config)

            if did_run not in [True, False]:
                raise RuntimeError(f"Test {test.name} did not return a boolean value.")

            if not did_run:
                match_not_run.append(test.name)
            else:
                match_run.append(test.name)
        else:
            not_match.append(test.name)

    if config.failures:
        # Convert the list of failed tests into a map: test name to the
        # number of failures (config.failures list may contain duplicates)
        failures_map = {}
        for test in config.failures:
            if test in failures_map:
                failures_map[test] += 1
            else:
                failures_map[test] = 1
        error_string = "The following tests failed:"
        for test, count in failures_map.items():
            error_string += f"\n   {test} ({count} times)."
        raise RuntimeError(error_string)

    if verbose:
        print(f"Tests that ran: {match_run}")
        print(f"Tests that matched but did not run: {match_not_run}")
        print(f"Tests that did not match: {not_match}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Testing AIE numerical correctness",
        description="This program compares numerical outputs on AIE against baseline values",
    )
    abs_path = lambda x: Path(x).absolute()
    parser.add_argument("output_dir", type=abs_path)
    parser.add_argument("iree_install_dir", type=abs_path)
    parser.add_argument("peano_install_dir", type=abs_path)
    parser.add_argument("--xrt-dir", type=abs_path)
    parser.add_argument("--vitis-dir", type=abs_path)
    parser.add_argument("--xrt_lite_n_core_rows", type=int)
    parser.add_argument("--xrt_lite_n_core_cols", type=int)
    parser.add_argument("--target_device", type=str, required=True)

    # TODO(newling) make bool options boolean, not integer (tried but had issues)
    parser.add_argument(
        "--return-on-fail",
        nargs="?",
        default=1,
        type=int,
        help=dedent(
            """
            If 0, then the script will continue running even if a test fails,
            enumerating all failures. Otherwise the script will exit on the first failure.
            """
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help=dedent(
            """
            Verbosity level. Currently
            0: total silence.
            1 (-v) : almost everything.
            2 (-vv) : everything.
            """
        ),
    )

    parser.add_argument(
        "--reset-npu-between-runs",
        action="store_true",
        help=(
            "If passed then the NPU is not reset between runs, otherwise it is reset. "
            "Resetting between runs can in theory help avoid certain types of "
            "errors in parts of the stack which these tests are not designed to catch."
        ),
    )

    parser.add_argument(
        "--do-not-run-aie",
        action="store_true",
        help=dedent(
            """
            If passed then the AIE backend will not be run. This is useful for
            ensuring that everything up to the AIE run and numerical comparison
            is working correctly, for example if you are not on a device with
            working AIE HW and runtime."
            """
        ),
    )

    tests = Tests()
    labels = tests.get_label_set()
    labels.append("All")
    names = tests.get_test_names()
    label_string = ", ".join(labels)
    name_string = ", ".join(names)

    help_string = (
        "A comma-separated list of test names or sets to run. Available test sets: "
        + f"{label_string}"
        + f". Available individual tests: {name_string}. "
    )

    parser.add_argument(
        "--tests",
        type=str,
        help=help_string,
        default="All",
    )

    parser.add_argument(
        "--additional-aie-compilation-flags",
        type=str,
        help=dedent(
            """
            Additional flags to pass to the AIE compiler, for all tests.
            Example, do print the IR between passes during compilation you might have:
            --additional_aie_compilation_flags="--mlir-print-ir-before-all --mlir-print-ir-module-scope
            --aie2xclbin-print-ir-before-all --aie2xclbin-print-ir-module-scope"'
            """
        ),
        default="",
    )

    parser.add_argument(
        "--device-hal",
        default="xrt-lite",
        const="xrt-lite",
        nargs="?",
        choices=["xrt", "xrt-lite"],
        help="device HAL to use (default: %(default)s)",
    )

    parser.add_argument(
        "--use_chess",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    test_set_list = args.tests.split(",")

    all_tests(
        tests,
        args.output_dir,
        args.iree_install_dir,
        args.peano_install_dir,
        args.xrt_dir,
        args.vitis_dir,
        args.return_on_fail,
        args.verbose,
        args.reset_npu_between_runs,
        args.do_not_run_aie,
        test_set_list,
        args.additional_aie_compilation_flags,
        args.device_hal,
        args.xrt_lite_n_core_rows,
        args.xrt_lite_n_core_cols,
        args.target_device,
        args.use_chess,
    )
