#!/usr/bin/env python3

# Copyright 2024 The IREE Authors

from abc import ABC, abstractmethod
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
)


def run_conv_test(config, aie_compilation_flags, filename, n_repeats):
    aie_vs_llvm_cpu(
        config,
        aie_compilation_flags,
        filename,
        tile_pipeline="conv-decompose",
        lower_to_aie_pipeline="objectFifo",
        n_repeats=n_repeats,
    )
    # Return True to indicate that the test ran.
    return True


class BaseTest(ABC):
    """
    Base class to be inherited by all tests. The derived instances will be
    created specifying the intended target device(s) they're to be run on; and
    will accordingly be `run` if the intended target device(s) contains
    the `target_device` found in `config`. The default set of targets to run on
    is the singleton set `["npu1_4col"]`.

    An instance of this class has a member `aie_compilation_flags`
    which are additional flags to be passed to the AIE backend compiler.

    Compilation flags can therefore be injected into tests in 2 ways:
    1) via the constructor of this base class
    2) via the `add_aie_compilation_flags` method
    """

    def __init__(self, run_on_target=["npu1_4col"], aie_compilation_flags=None):
        self.run_on_target = [] if run_on_target is None else run_on_target
        self.aie_compilation_flags = (
            [] if aie_compilation_flags is None else aie_compilation_flags
        )

        # NB: derived classes should add labels to this list in their
        # constructor, never overwrite it.
        self.labels = ["All"]

    def add_aie_compilation_flags(self, flags):
        if flags:
            self.aie_compilation_flags += flags
            # unique-ify the list
            self.aie_compilation_flags = list(set(self.aie_compilation_flags))

    def run(self, config):
        if config.target_device in self.run_on_target:
            return self._execute(config)
        return False

    @abstractmethod
    def _execute(self, config):
        raise RuntimeError("Derived class must implement this method")


class ConvolutionFromTemplate(BaseTest):
    def __init__(self, params):
        super().__init__()
        self.generator = ConvolutionMlirGenerator(**params)
        params = self.generator.params
        conv_type = params["conv_type"]
        N = params["N"]
        IW = params["IW"]
        in_type = params["input_element_type"]
        out_type = params["output_element_type"]
        # TODO(newling) Use all parameters in name, to avoid name collision.
        self.name = f"{conv_type}_{N}_{IW}_{in_type}_{out_type}"
        self.labels += ["Convolution"]

    def _execute(self, config):
        # Generate MLIR file:
        output_dir = config.output_dir
        filename = output_dir / f"{self.name}.mlir"
        self.generator.write_to_file(filename)
        # Perform numerical comparison between AIE and CPU:
        return run_conv_test(config, self.aie_compilation_flags, filename, n_repeats=2)


class ConvolutionNHWCQ(BaseTest):
    def __init__(self):
        super().__init__()
        self.name = "convolution_nhwc_q"
        self.labels += ["Convolution", "ConvolutionNHWCQ"]

    def _execute(self, config):
        files_dir = config.file_dir / "test_files"
        filename = files_dir / "conv2d_nhwc_q.mlir"
        return run_conv_test(config, self.aie_compilation_flags, filename, n_repeats=1)


class MultipleDispatches(BaseTest):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.labels += ["Matmul", "MultipleDispatches"]

    def _execute(self, config):
        test_files_dir = config.file_dir / "test_files"
        self.filename = test_files_dir / f"{self.name}.mlir"
        # TODO(newling) did Maks ever document why this is here, if so add an
        # explainer.
        if config.xdna_datetime and config.xdna_datetime < 20240801:
            aie_vs_llvm_cpu(
                config,
                self.aie_compilation_flags,
                self.filename,
                function_name="three_$mm$",
            )
            return True
        else:
            # Return False to indicate that the test did not run.
            return False


class BaseMatmul(BaseTest):
    def __init__(
        self,
        run_on_target,
        aie_compilation_flags,
        M,
        N,
        K,
        input_type,
        acc_type,
        use_ukernel=False,
        lower_to_aie_pipeline="objectFifo",
        tile_pipeline="pack-peel",
        n_repeats=1,
    ):
        super().__init__(run_on_target, aie_compilation_flags)
        self.M = M
        self.N = N
        self.K = K
        self.input_type = input_type
        self.acc_type = acc_type
        self.tile_pipeline = tile_pipeline
        self.lower_to_aie_pipeline = lower_to_aie_pipeline
        self.use_ukernel = use_ukernel
        self.n_repeats = n_repeats
        self.labels.append("Matmul")
        if use_ukernel:
            self.labels.append("UKernel")

    def vs_cpu(self, config, filename):
        if self.use_ukernel and not config.vitis_dir:
            return False

        aie_vs_llvm_cpu(
            config=config,
            aie_compilation_flags=self.aie_compilation_flags,
            test_file=filename,
            use_ukernel=self.use_ukernel,
            tile_pipeline=self.tile_pipeline,
            lower_to_aie_pipeline=self.lower_to_aie_pipeline,
            n_repeats=self.n_repeats,
        )

        return True

    def generate(self, filename, template_name):
        generate_matmul_test(
            filename,
            template_name,
            self.M,
            self.N,
            self.K,
            self.input_type,
            self.acc_type,
        )


class MatmulFullBias(BaseMatmul):
    """
    A test of the form matmul(A,B) + C where A:MxK, B:KxN, C:MxN
    """

    def __init__(self, M, N, K, input_type, acc_type, run_on_target=["npu1_4col"]):
        super().__init__(
            run_on_target=run_on_target,
            aie_compilation_flags=None,
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
            lower_to_aie_pipeline="air",
        )
        self.name = f"matmul_full_bias_{M}_{N}_{K}_{input_type}_{acc_type}"
        self.labels.append("MatmulFullBias")

    def _execute(self, config):
        filename = config.output_dir / f"{self.name}.mlir"
        matmul_template_dir = config.file_dir / "matmul_template"
        template_name = matmul_template_dir / "matmul_bias_MxK_KxN_MxN.mlir"
        self.generate(filename, template_name)
        self.vs_cpu(config, filename)
        return True


class VanillaMatmul(BaseMatmul):
    """
    A test of the form matmul(A,B) where A:MxK, B:KxN
    """

    def __init__(
        self,
        M,
        N,
        K,
        input_type,
        acc_type,
        name_suffix="",
        use_ukernel=False,
        run_on_target=["npu1_4col"],
        additional_labels=None,
        aie_compilation_flags=None,
    ):
        super().__init__(
            run_on_target=run_on_target,
            aie_compilation_flags=aie_compilation_flags,
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
            tile_pipeline="pack-peel",
            use_ukernel=use_ukernel,
            n_repeats=1,
        )

        self.name = f"vanilla_matmul_{M}_{N}_{K}_{input_type}_{acc_type}"
        if name_suffix:
            self.name += f"_{name_suffix}"
        if use_ukernel:
            self.name += "_ukernel"
        self.labels.append("VanillaMatmul")
        if additional_labels:
            self.labels += additional_labels
        self.use_ukernel = use_ukernel

    def _execute(self, config):
        self.filename = config.output_dir / f"{self.name}.mlir"
        matmul_template_dir = config.file_dir / "matmul_template"
        template_name = matmul_template_dir / "matmul_MxK_KxN.mlir"
        self.generate(self.filename, template_name)
        self.vs_cpu(config, self.filename)

        return True


class MatmulThinBias(BaseMatmul):
    """
    A test of the form matmul(A,B) + C where A:MxK, B:KxN, C:N
    """

    def __init__(
        self,
        M,
        N,
        K,
        input_type,
        acc_type,
        use_ukernel=False,
        run_on_target=["npu1_4col"],
    ):
        super().__init__(
            run_on_target=run_on_target,
            aie_compilation_flags=None,
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
            lower_to_aie_pipeline="air",
            use_ukernel=use_ukernel,
        )

        self.name = f"matmul_thin_bias_{M}_{N}_{K}_{input_type}_{acc_type}"
        if use_ukernel:
            self.name += "_ukernel"
        self.labels.append("MatmulThinBias")

    def _execute(self, config):
        self.filename = config.output_dir / f"{self.name}.mlir"
        matmul_template_dir = config.file_dir / "matmul_template"
        template_name = matmul_template_dir / "matmul_bias_MxK_KxN_N.mlir"
        self.generate(self.filename, template_name)
        return self.vs_cpu(config, self.filename)


class BatchMatmul(BaseMatmul):
    """
    A test of the form batch_matmul(A,B) where A:BxMxK, B:BxKxN
    """

    def __init__(self, B, M, N, K, input_type, acc_type, run_on_target=["npu1_4col"]):
        super().__init__(
            run_on_target=run_on_target,
            aie_compilation_flags=None,
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
            tile_pipeline="pack-peel",
            n_repeats=1,
        )

        self.name = f"batch_matmul_{B}_{M}_{N}_{K}_{input_type}_{acc_type}"
        self.labels.append("BatchMatmul")
        self.B = B

    def _execute(self, config):
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
        return self.vs_cpu(config, self.filename)


class MatmulTruncf(BaseMatmul):
    """
    A test of the form matmul(A,B) + truncf(C) where A:MxK, B:KxM and C:MxM
    """

    def __init__(
        self,
        M,
        K,
        input_type,
        acc_type,
        lhs,
        rhs,
        expected_out,
        run_on_target=["npu1_4col"],
    ):
        super().__init__(
            run_on_target=run_on_target,
            aie_compilation_flags=None,
            M=M,
            N=M,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
            tile_pipeline="pack-peel",
            n_repeats=1,
        )

        # Assertions on shapes: Check that lhs is MxK, rhs is KxM, and expected_out is MxM
        assert lhs.shape == (M, K)
        assert rhs.shape == (K, M)
        assert expected_out.shape == (M, M)

        self.name = f"matmul_truncf_{M}_{K}_{input_type}_{acc_type}"
        self.labels.append("MatmulTruncf")
        self.lhs = lhs
        self.rhs = rhs
        self.expected_out = expected_out

    def _execute(self, config):
        self.filename = config.output_dir / f"{self.name}.mlir"
        matmul_template_dir = config.file_dir / "matmul_template"
        template_name = matmul_template_dir / "matmul_truncf_MxK_KxN.mlir"
        self.generate(self.filename, template_name)

        input_args = generate_inputs(
            self.filename, config.output_dir, 1, {1: self.lhs, 2: self.rhs}
        )
        """
        currently without enabling loop coalescing and unit dimension collapsing
        we run out of program memory, this is under investigation. We also 
        enable function outlining.
        """
        self.aie_compilation_flags += [
            "--iree-amdaie-enable-coalescing-loops",  # enable loop coalescing
            "--iree-amdaie-enable-collapsing-unit-dims",  # enable unit dimension collapsing
            "--iree-amdaie-enable-function-outlining",  # enable function outlining
        ]
        aie_vs_baseline(
            config=config,
            aie_compilation_flags=self.aie_compilation_flags,
            test_file=self.filename,
            input_args=input_args,
            baseline_value=self.expected_out,
            use_ukernel=self.use_ukernel,
            tile_pipeline=self.tile_pipeline,
            function_name=None,
            seed=1,
            rtol=0,
            atol=0,
            lower_to_aie_pipeline=self.lower_to_aie_pipeline,
            n_repeats=self.n_repeats,
            output_type=get_output_type(self.filename),
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
    aie_compilation_flags,
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

    additional_flags = aie_compilation_flags

    aie_compilation_flags = [
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
        aie_compilation_flags += ["--iree-amd-aie-show-invoked-commands"]

    if use_ukernel:
        aie_compilation_flags += ["--iree-amdaie-enable-ukernels=all"]

    for additional_flag in additional_flags:
        if additional_flag not in aie_compilation_flags:
            aie_compilation_flags += [additional_flag]

    aie_compilation_flags += [
        "-o",
        config.output_dir / f"{name}_aie.vmfb",
    ]

    start = time.monotonic_ns()
    shell_out(aie_compilation_flags, config.output_dir, config.verbose)
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
    aie_compilation_flags = [
        config.iree_compile_exe,
        test_file,
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu-features=host",
        "-o",
        f"{cpu_vmfb}",
    ]
    shell_out(aie_compilation_flags, workdir=config.output_dir, verbose=config.verbose)

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
    Global state used for all tests. Stores paths to executables used.
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
        reset_npu_between_runs,
        do_not_run_aie,
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
        self.verbose = verbose
        self.xdna_datetime = None
        self.xdna_hash = None
        self.reset_npu_between_runs = reset_npu_between_runs
        self.do_not_run_aie = do_not_run_aie
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

    def add_aie_compilation_flags(self, flag):
        self.aie_compilation_flags += flag

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
    aie_compilation_flags,
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
    collapse_unit_dims=False,
    function_outline=False,
):
    """
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
    collapse_unit_dims:
        Whether to enable collapsing of unit dimensions when compiling for AIE backend
    function_outline:
        Whether to enable linalg function outlining when compiling for AIE backend
    """

    name = name_from_mlir_filename(test_file)

    aie_vmfb = generate_aie_vmfb(
        config,
        aie_compilation_flags,
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

        summary_string = compare(baseline_value, aie_output, rtol, atol)
        if summary_string:
            print(summary_string)
            raise RuntimeError("Test failed, exiting.")


def aie_vs_llvm_cpu(
    config,
    aie_compilation_flags,
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
        aie_compilation_flags,
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
    )


class Tests:

    def add_aie_compilation_flags(self, flags):
        for test in self.tests:
            test.add_aie_compilation_flags(flags)

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
        self.register(
            MatmulTruncf(
                8,
                8,
                "bf16",
                "f32",
                101 * np.ones([8, 8]),
                3 * np.eye(8),
                302 * np.ones([8, 8]),
            )
        )

        self.register(
            MatmulTruncf(
                128,
                256,
                "bf16",
                "f32",
                2 * np.ones([128, 256]),
                3 * np.ones([256, 128]),
                1536 * np.ones([128, 128]),
            )
        )

        # BatchMatmul test(s):
        for input_type, acc_type in zip(["i32", "bf16"], ["i32", "f32"]):
            # Batch size = 1:
            self.register(BatchMatmul(1, 128, 128, 256, input_type, acc_type))
            # Batch size = 2:
            self.register(BatchMatmul(2, 64, 64, 64, input_type, acc_type))

        # MatmulThinBias test(s):
        self.register(MatmulThinBias(1024, 1024, 512, "bf16", "f32", use_ukernel=True))
        self.register(MatmulThinBias(1024, 1024, 512, "bf16", "f32"))

        # VanillaMatmul test(s):
        self.register(
            VanillaMatmul(
                32,
                32,
                32,
                "i32",
                "i32",
                run_on_target=["npu1_4col", "npu4"],
            )
        )
        self.register(
            VanillaMatmul(
                32,
                32,
                32,
                "i32",
                "i32",
                name_suffix="infinite_loop",
                run_on_target=["npu1_4col", "npu4"],
                aie_compilation_flags=[
                    "--iree-amdaie-enable-infinite-loop-around-core-block=true"
                ],
            )
        )
        self.register(VanillaMatmul(32, 32, 64, "bf16", "f32"))

        # TODO: Failure is expected for the 128x128 case we don't yet understand why.
        self.register(
            VanillaMatmul(
                64,
                64,
                64,
                "bf16",
                "f32",
                use_ukernel=True,
                run_on_target=["npu4"],
            )
        )

        self.register(
            VanillaMatmul(
                512,
                512,
                4096,
                "bf16",
                "f32",
                additional_labels=["Performance"],
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
    verbose,
    reset_npu_between_runs,
    do_not_run_aie,
    test_set,
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
        reset_npu_between_runs,
        do_not_run_aie,
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
        # match = "All" in test_set
        match = test.name in test_set
        for label in test.labels:
            match = match or label in test_set

        if match:
            did_run = test.run(config)
            if not did_run:
                match_not_run.append(test.name)
            else:
                match_run.append(test.name)
        else:
            not_match.append(test.name)

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
    parser.add_argument(
        "--xrt_lite_n_core_rows",
        type=int,
        help="Number of AIE core rows of the xrt-lite device to use",
    )
    parser.add_argument(
        "--xrt_lite_n_core_cols",
        type=int,
        help="Number of AIE core columns of the xrt-lite device to use",
    )

    # Taken from AMDAIEEnums.td
    current_devices = [
        "xcvc1902",
        "xcve2302",
        "xcve2802",
        "npu1",
        "npu1_1col",
        "npu1_2col",
        "npu1_3col",
        "npu1_4col",
        "npu4",
    ]
    target_device_help_string = f"Target device to run the tests on. Available options: {current_devices}. Hint: phoenix devices start with 'npu1' and strix devices start with 'npu4'."

    parser.add_argument(
        "--target_device", type=str, required=True, help=target_device_help_string
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

    parser.add_argument(
        "--aie-compilation-flags",
        type=str,
        help=dedent(
            """
            Additional flags to pass to the AIE compiler, for all tests.
            Example, do print the IR between passes during compilation you might have:
            --aie_compilation_flags="--mlir-print-ir-before-all --mlir-print-ir-module-scope
            --aie2xclbin-print-ir-before-all --aie2xclbin-print-ir-module-scope"'
            """
        ),
        default="",
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

    parser.epilog = "Example call: ./run.py --verbose  output_dir ${IREE_INSTALL}  ${LLVM_AIE}  --target_device=npu1_4col --xrt_lite_n_core_rows=4 --xrt_lite_n_core_cols=4 --tests=Matmul"

    args = parser.parse_args()

    test_set_list = args.tests.split(",")

    if args.target_device not in current_devices:
        raise ValueError(
            f"Invalid target device '{args.target_device}'. Available options: {current_devices}"
        )
    tests.add_aie_compilation_flags(args.aie_compilation_flags)

    all_tests(
        tests,
        args.output_dir,
        args.iree_install_dir,
        args.peano_install_dir,
        args.xrt_dir,
        args.vitis_dir,
        args.verbose,
        args.reset_npu_between_runs,
        args.do_not_run_aie,
        test_set_list,
        args.device_hal,
        args.xrt_lite_n_core_rows,
        args.xrt_lite_n_core_cols,
        args.target_device,
        args.use_chess,
    )
