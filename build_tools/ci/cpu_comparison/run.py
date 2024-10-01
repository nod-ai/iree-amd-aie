#!/usr/bin/env python3

# Copyright 2024 The IREE Authors

import sys
import argparse
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from textwrap import dedent

import numpy as np

from input_generator import generate_inputs, verify_determinism, load_input
from matmul_template.matmul_generator import generate_matmul_test
from convolution_template.convolution_generator import ConvolutionMlirGenerator
from output_comparer import compare


def matmul_from_input_strings(input_args):
    """
    Input 'input_args' should be a list with two strings, of the form

    ["--input=3x40xf32=@<binary_file>", "input=2x2xi32=@<binary_file>"]

    where the binary files contain the input matrices.

    This function
    1) loads the input matrices from the binary files
    2) returns the result of multiplying the matrices together
    """
    if len(input_args) != 2:
        raise RuntimeError(f"Expected 2 arguments, got {input_args=}")
    a = load_input(input_args[0])
    b = load_input(input_args[1])
    return np.matmul(a, b)


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
            file=sys.stderr
        )
    if raise_on_error and handle.returncode != 0:
        raise RuntimeError(
            f"Error executing script, error code was {handle.returncode}",
            file=sys.stderr
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
    Compile a test file for IREE's AIE backend, returning path to the compiled
    module.
    """

    additional_flags = config.additional_aie_compilation_flags.split()

    compilation_flags = [
        config.iree_compile_exe,
        test_file,
        "--iree-hal-target-backends=amd-aie",
        f"--iree-amdaie-tile-pipeline={tile_pipeline}",
        f"--iree-amdaie-lower-to-aie-pipeline={lower_to_aie_pipeline}",
        "--iree-amdaie-matmul-elementwise-fusion",
        f"--iree-amd-aie-peano-install-dir={config.peano_dir}",
        f"--iree-amd-aie-install-dir={config.iree_install_dir}",
        f"--iree-amd-aie-vitis-install-dir={config.vitis_dir}",
        f"--iree-hal-dump-executable-files-to={config.output_dir}",
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


def generate_aie_output(config, aie_vmfb, input_args, function_name, name):
    """
    Run a compiled AIE module (aie_vmfb), returning a numpy array of the output.
    """

    aie_npy = config.output_dir / f"{name}_aie.npy"
    run_args = [
        config.iree_run_exe,
        f"--module={aie_vmfb}",
        *input_args,
        "--device=xrt",
        f"--output=@{aie_npy}",
    ]
    if function_name:
        run_args += [f"--function={function_name}"]
    if config.reset_npu_between_runs:
        shell_out(config.reset_npu_script, verbose=config.verbose)

    start = time.monotonic_ns()
    shell_out(run_args, config.output_dir, config.verbose)
    run_time = time.monotonic_ns() - start

    if config.verbose:
        print(f"Time spent in running the model: {run_time // 1e6} [ms]")

    return np.load(aie_npy)


def generate_llvm_cpu_output(
    config,
    name,
    test_file,
    input_args,
    function_name,
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

    cpu_npy = config.output_dir / f"{name}_cpu.npy"
    run_args = [
        config.iree_run_exe,
        f"--module={cpu_vmfb}",
        *input_args,
        f"--output=@{cpu_npy}",
    ]
    if function_name:
        run_args += [f"--function={function_name}"]
    shell_out(run_args, workdir=config.output_dir, verbose=config.verbose)
    return np.load(cpu_npy)


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
        return_on_fail,
        reset_npu_between_runs,
        do_not_run_aie,
        additional_aie_compilation_flags,
    ):
        self.output_dir = output_dir
        self.iree_install_dir = iree_install_dir
        self.peano_dir = peano_dir
        self.xrt_dir = xrt_dir
        self.vitis_dir = vitis_dir
        self.file_dir = file_dir
        self.iree_compile_exe = iree_compile_exe
        self.iree_run_exe = iree_run_exe
        self.return_on_fail = return_on_fail
        self.verbose = verbose
        self.xdna_datetime = None
        self.xdna_hash = None
        self.reset_npu_between_runs = reset_npu_between_runs
        self.do_not_run_aie = do_not_run_aie
        self.additional_aie_compilation_flags = additional_aie_compilation_flags

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
        return_on_fail:       {self.return_on_fail}
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
    pipeline:
        The tiling pipeline to use when compiling for the AIE backend
    function_name:
        The name of the function to run (the test file may contain multiple
        functions)
    ...
    n_repeats:
        The number of times to run the test. This is useful for tests which
        may pass only sometimes due to driver issues, etc.
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
        )

        summary_string = compare(baseline_value, aie_output, rtol, atol)
        if summary_string:
            print(summary_string)
            config.failures.append(test_file)
            if config.return_on_fail:
                raise RuntimeError("Test failed, exiting.")


def aie_vs_llvm_cpu(
    config,
    test_file,
    use_ukernel=False,
    tile_pipeline="pad-pack",
    lower_to_aie_pipeline="air",
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

    cpu_output = generate_llvm_cpu_output(
        config, name, test_file, input_args, function_name
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
    )


def aie_vs_np_matmul(
    config,
    test_file,
    use_ukernel=False,
    tile_pipeline="pad-pack",
    lower_to_aie_pipeline="air",
    function_name=None,
    seed=1,
    rtol=1e-6,
    atol=1e-6,
    n_repeats=1,
):
    """ """

    if n_repeats == 0:
        return

    name = name_from_mlir_filename(test_file)
    input_args = generate_inputs(test_file, config.output_dir, seed)
    numpy_output = matmul_from_input_strings(input_args)
    aie_vs_baseline(
        config,
        test_file,
        input_args,
        numpy_output,
        use_ukernel,
        tile_pipeline,
        lower_to_aie_pipeline,
        function_name,
        seed,
        rtol,
        atol,
        n_repeats,
    )


class TestSet:
    def __init__(self, name):
        self.name = name

    def run(self, config):
        raise NotImplementedError("Subclasses must implement this method")


class ConvolutionTemplateSet(TestSet):
    def __init__(self):
        super().__init__("ConvolutionTemplate")

    def run(self, config):

        testSet = [
            {
                "conv_type": "conv_2d_nhwc_hwcf",
                "N": 2,
                "IH": 14,
                "IC": 32,
                "OC": 64,
                "KH": 3,
                "input_element_type": "i32",
                "output_element_type": "i32",
            },
            {
                "conv_type": "conv_2d_nhwc_hwcf",
                "N": 2,
                "IH": 14,
                "IC": 32,
                "OC": 64,
                "KH": 3,
                "input_element_type": "bf16",
                "output_element_type": "f32",
            },
            {
                "conv_type": "conv_2d_nhwc_hwcf",
                "N": 2,
                "IH": 14,
                "IC": 32,
                "OC": 64,
                "KH": 3,
                "input_element_type": "i8",
                "output_element_type": "i32",
            },
            {
                "conv_type": "depthwise_conv_2d_nhwc_hwc",
                "N": 1,
                "IH": 14,
                "IC": 64,
                "KH": 3,
                "input_element_type": "i32",
                "output_element_type": "i32",
            },
        ]

        output_dir = config.output_dir
        test_name = output_dir / "test_from_template.mlir"
        for testMap in testSet:
            convGen = ConvolutionMlirGenerator(**testMap)
            convGen.write_to_file(test_name)
            n_conv_repeats = 4

            aie_vs_llvm_cpu(
                config,
                test_name,
                tile_pipeline="conv-decompose",
                lower_to_aie_pipeline="air",
                n_repeats=n_conv_repeats,
            )


class ConvolutionSet(TestSet):
    def __init__(self):
        super().__init__("Convolution")

    def run(self, config):
        test_files_dir = config.file_dir / "test_files"

        for name in [
            "conv2d_nhwc_q",
        ]:
            n_conv_repeats = 2
            aie_vs_llvm_cpu(
                config,
                test_files_dir / f"{name}.mlir",
                tile_pipeline="conv-decompose",
                lower_to_aie_pipeline="air",
                n_repeats=n_conv_repeats,
            )


class MatmulSet(TestSet):
    def __init__(self):
        super().__init__("Matmul")

    def run(self, config):
        matmul_template_dir = config.file_dir / "matmul_template"
        test_files_dir = config.file_dir / "test_files"
        output_dir = config.output_dir

        # Test(s) of the form matmul(A,B) + C where A:MxK, B:KxN, C:MxN
        test_name = output_dir / "test_from_template_full_bias.mlir"
        template_name = matmul_template_dir / "matmul_bias_MxK_KxN_MxN.mlir"
        generate_matmul_test(test_name, template_name, 128, 128, 256, "i32", "i32")
        aie_vs_llvm_cpu(
            config,
            test_name,
            tile_pipeline="pack-peel",
            lower_to_aie_pipeline="air",
            rtol=0,
            atol=0,
        )

        if config.xdna_datetime and config.xdna_datetime < 20240801:
            for name in [
                "two_matmul_switching",
                "matmul_f32_8_8_4",
                "matmul_f32_8_4_8",
            ]:
                aie_vs_llvm_cpu(config, test_files_dir / f"{name}.mlir")

            aie_vs_llvm_cpu(
                config,
                test_files_dir / "three_matmuls.mlir",
                function_name="three_$mm$",
            )

        # Test(s) of the form matmul(A,B) where A:MxK, B:KxN
        test_name = output_dir / "test_from_template.mlir"
        template_name = matmul_template_dir / "matmul_MxK_KxN.mlir"
        generate_matmul_test(test_name, template_name, 32, 32, 64, "bf16", "f32")
        aie_vs_llvm_cpu(config, test_name)

        # Test(s) of the form matmul(A,B) + C where A:MxK, B:KxN, C:N
        test_name = output_dir / "test_from_template_bias_N.mlir"
        template_name = matmul_template_dir / "matmul_bias_MxK_KxN_N.mlir"
        generate_matmul_test(test_name, template_name, 1024, 1024, 512, "bf16", "f32")
        if config.vitis_dir:
            aie_vs_llvm_cpu(
                config,
                test_name,
                tile_pipeline="pack-peel",
                lower_to_aie_pipeline="air",
                use_ukernel=True,
            )
        aie_vs_llvm_cpu(
            config,
            test_name,
            tile_pipeline="pack-peel",
            lower_to_aie_pipeline="air",
            use_ukernel=False,
        )

        # Test(s) of the form batch_matmul(A,B) where A:BxMxK, B:BxKxN
        template_name = matmul_template_dir / "batch_matmul_BxMxK_BxKxN.mlir"
        for lhs_type, acc_type in zip(["i32", "bf16"], ["i32", "f32"]):
            # Batch size = 1
            test_name = (
                output_dir / f"test_from_template_bmm_1_{lhs_type}_{acc_type}.mlir"
            )
            generate_matmul_test(
                test_name, template_name, 128, 128, 256, lhs_type, acc_type, b=1
            )
            aie_vs_llvm_cpu(
                config,
                test_name,
                tile_pipeline="pack-peel",
                lower_to_aie_pipeline="objectFifo",
            )
            # Batch size = 2
            test_name = (
                output_dir / f"test_from_template_bmm_2_{lhs_type}_{acc_type}.mlir"
            )
            generate_matmul_test(
                test_name, template_name, 64, 64, 64, lhs_type, acc_type, b=2
            )
            aie_vs_llvm_cpu(
                config,
                test_name,
                tile_pipeline="pack-peel",
                lower_to_aie_pipeline="objectFifo",
            )


class SmokeSet(TestSet):
    def __init__(self):
        super().__init__("Smoke")

    def run(self, config):
        file_dir = config.file_dir
        output_dir = config.output_dir

        # The most basic test, direct from .mlir file using all defaults
        test_files_dir = file_dir / "test_files"
        aie_vs_llvm_cpu(config, test_files_dir / "matmul_int32.mlir")

        # Using objectFifo pipeline
        test_name = output_dir / "test_from_template.mlir"
        matmul_template_dir = file_dir / "matmul_template"
        test_name = output_dir / "test_from_objectfifo_basic.mlir"
        template_name = matmul_template_dir / "matmul_MxK_KxN.mlir"
        generate_matmul_test(test_name, template_name, 64, 64, 64, "bf16", "f32")
        aie_vs_llvm_cpu(
            config,
            test_name,
            tile_pipeline="pack-peel",
            lower_to_aie_pipeline="objectFifo",
        )


def get_test_partition():
    return [ConvolutionTemplateSet(), ConvolutionSet(), MatmulSet(), SmokeSet()]


def all_tests(
    output_dir,
    iree_install_dir,
    peano_dir,
    xrt_dir,
    vitis_dir,
    return_on_fail,
    verbose,
    reset_npu_between_runs,
    do_not_run_aie,
    test_set,
    additional_aie_compilation_flags,
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
        return_on_fail,
        reset_npu_between_runs,
        do_not_run_aie,
        additional_aie_compilation_flags,
    )
    if verbose:
        print(config)

    # Sanity check that results are reproducible across platforms
    verify_determinism()

    # Verify a very basic script runs before running the more complex tests
    if platform.system() != "Windows":
        shell_out(["pwd"], verbose=config.verbose)

    partition = get_test_partition()
    partition_names = [p.name for p in partition]
    map_to_partition = {p.name: p for p in partition}
    if "All" in test_set:
        test_set = partition_names

    # For each test in test_set, find the partition it belongs to and run it
    # if no partition is found, raise error.
    for test in test_set:
        if test not in partition_names:
            errorMessage = f"Test set '{test}' not found in available test sets. The available test sets are:"
            for name in partition_names:
                errorMessage += f"\n  {name}"
            raise ValueError(errorMessage)
        partition = map_to_partition[test]
        partition.run(config)

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
            If passed, then the AIE backend will not be run. This is useful for
            ensuring that everything up to the AIE run and numerical comparison
            is working correctly, for example if you are not on a device with
            working AIE HW and runtime."
            """
        ),
    )

    partition = get_test_partition()
    partition_names = [p.name for p in partition]
    partition_names_and_all = partition_names + ["All"]
    help_string = (
        "A comma-separated list of test sets. Available test sets are: "
        + ", ".join(partition_names_and_all)
    )

    parser.add_argument(
        "--test-set",
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

    args = parser.parse_args()

    test_set_list = args.test_set.split(",")
    all_tests(
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
    )
