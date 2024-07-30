#!/usr/bin/env python3

# Copyright 2024 The IREE Authors

import argparse
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from textwrap import dedent

import numpy as np

from input_generator import generate_inputs, verify_determinism
from matmul_template.matmul_generator import generate_matmul_test
from input_generator import generate_inputs, verify_determinism, load_input
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
        raise RuntimeError(f"Expected 2 arguments, got {len(ab)}")
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
    for directory in search_dirs:
        executable_path = directory / executable_name
        if executable_path.is_file():
            return executable_path
    raise RuntimeError(
        f"No '{executable_name}' executable found in '{install_dir}' or subdirectories."
    )


def shell_out(cmd: list, workdir=None, verbose=False):
    if workdir is None:
        workdir = Path.cwd()
    if not isinstance(cmd, list):
        cmd = [cmd]
    for i, c in enumerate(cmd):
        if isinstance(c, Path):
            cmd[i] = str(c)
    env = os.environ
    if verbose:
        _cmd = " ".join([f"{k}={v}" for k, v in env.items()]) + " " + " ".join(cmd)
        print(f"Running the following command:\n{_cmd}")

    handle = subprocess.run(cmd, capture_output=True, cwd=workdir, env=env)
    stderr_decode = handle.stderr.decode("utf-8").strip()
    stdout_decode = handle.stdout.decode("utf-8").strip()
    if verbose:
        if stdout_decode:
            print("Standard output from script:")
            print(stdout_decode)
        if stderr_decode:
            print("Standard error from script:")
            print(stderr_decode)
    if handle.returncode != 0:
        raise RuntimeError(
            f"Error executing script, error code was {handle.returncode}"
        )
    return stdout_decode, stderr_decode


def generate_aie_output(
    config,
    name,
    pipeline,
    use_ukernel,
    test_file,
    input_args,
    function_name,
):
    """
    Compile and run a test file for AIE, returning a numpy array of the output.
    """

    compilation_flags = [
        config.iree_compile_exe,
        test_file,
        "--iree-hal-target-backends=amd-aie",
        f"--iree-amdaie-tile-pipeline={pipeline}",
        "--iree-amdaie-matmul-elementwise-fusion",
        f"--iree-amd-aie-peano-install-dir={config.peano_dir}",
        f"--iree-amd-aie-install-dir={config.iree_install_dir}",
        f"--iree-amd-aie-vitis-install-dir={config.vitis_dir}",
        f"--iree-hal-dump-executable-files-to={config.output_dir}",
        "--iree-scheduling-optimize-bindings=false",
        f"--mlir-disable-threading",
        "--mlir-elide-resource-strings-if-larger=10",
        "-o",
        config.output_dir / f"{name}_aie.vmfb",
    ]
    if config.verbose:
        compilation_flags += ["--iree-amd-aie-show-invoked-commands"]

    if use_ukernel:
        MM_KERNEL_URL = (
            "https://github.com/nod-ai/iree-amd-aie/releases/download/ukernels/mm.o"
        )
        compilation_flags += ["--iree-amdaie-enable-ukernels=all"]
        mm_fn = config.output_dir / "mm.o"
        if mm_fn.exists():
            if config.verbose:
                print(f"File {mm_fn} already exists")
        else:
            if config.verbose:
                print(f"Attempting to download {MM_KERNEL_URL} to {mm_fn}")

            urllib.request.urlretrieve(MM_KERNEL_URL, mm_fn)
            if not mm_fn.exists():
                raise RuntimeError("Failed to download mm.o")

    start = time.monotonic_ns()
    shell_out(compilation_flags, config.output_dir, config.verbose)
    compile_time = time.monotonic_ns() - start

    aie_vmfb = config.output_dir / f"{name}_aie.vmfb"
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
        shell_out(config.reset_npu_script)
    start = time.monotonic_ns()
    shell_out(run_args, config.output_dir, config.verbose)
    run_time = time.monotonic_ns() - start

    print(f"Time spent in compilation: {compile_time // 1e6} [ms]")
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
        self.reset_npu_between_runs = reset_npu_between_runs

        # Try get the xrt and (linux) kernel versions.
        self.linux_kernel = "undetermined"
        self.xrt_hash_date = "undetermined"
        self.xrt_hash = "undetermined"
        self.xrt_release = "undetermined"
        self.peano_commit_hash = "undetermined"
        xrt_bin_dir = xrt_dir / "bin"
        xrt_smi_exe = xrt_bin_dir / "xrt-smi"
        if not xrt_smi_exe.exists():
            xrt_smi_exe = xrt_bin_dir / "xbutil"
        if not xrt_smi_exe.exists():
            raise RuntimeError(f"Neither xrt-smi nor xbutil found in {xrt_bin_dir}")

        self.reset_npu_script = file_dir.parent / "reset_npu.sh"
        if reset_npu_between_runs and not self.reset_npu_script.exists():
            raise RuntimeError(
                f"The file {self.reset_npu_script} does not exist, and reset_npu_script=True"
            )

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

        # no clue why but peano clang dumps -v to stderr
        _, clang_v_output = shell_out([peano_dir / "bin" / "clang", "-v"])
        peano_commit_hash = re.findall(
            r"clang version \d+\.\d+\.\d+ \(https://github.com/Xilinx/llvm-aie (\w+)\)",
            clang_v_output,
            re.MULTILINE,
        )
        if peano_commit_hash:
            self.peano_commit_hash = peano_commit_hash[0]
        else:
            self.peano_commit_hash = "undetermined"

        # Populated at runtime
        self.failures = []

        if not isinstance(self.verbose, bool) and not isinstance(self.verbose, int):
            raise ValueError(
                f"verbose must be a boolean or integer, not {type(verbose)}"
            )

    def __str__(self):
        return dedent(
            f"""
        Settings and versions used in all tests
        -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
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
        -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

        Some information on the above settings / versions
        =================================================
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
    use_ukernel=False,
    pipeline="pad-pack",
    function_name=None,
    seed=1,
    rtol=1e-6,
    atol=1e-6,
    n_repeats=1,
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

    name = Path(test_file).name.replace(".mlir", "")

    for i in range(n_repeats):
        if config.verbose:
            print(f"Run #{i + 1} of {n_repeats} for {test_file}")

        aie_output = generate_aie_output(
            config,
            name,
            pipeline,
            use_ukernel,
            test_file,
            input_args,
            function_name,
        )

        same_result = compare(baseline_value, aie_output, rtol, atol)
        if not same_result:
            config.failures.append(test_file)
            if config.return_on_fail:
                raise RuntimeError("Test failed, exiting.")


def aie_vs_llvm_cpu(
    config,
    test_file,
    use_ukernel=False,
    pipeline="pad-pack",
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
        pipeline,
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
    pipeline="pad-pack",
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
        pipeline,
        function_name,
        seed,
        rtol,
        atol,
        n_repeats,
    )


def all_tests(
    output_dir,
    iree_install_dir,
    peano_dir,
    xrt_dir,
    vitis_dir,
    return_on_fail,
    verbose,
    reset_npu_between_runs,
):
    """
    There are a few ways to add tests to this function:

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
    )
    if verbose:
        print(config)

    # Sanity check that results are reproducible across platforms
    verify_determinism()

    # Verify a very basic script runs before running the more complex tests
    shell_out(["pwd"], verbose=config.verbose)

    test_files_dir = file_dir / "test_files"
    aie_vs_np_matmul(config, test_files_dir / "matmul_int32.mlir")
    for name in [
        "matmul_int32",
        "two_matmul_switching",
        "matmul_f32_8_8_4",
        "matmul_f32_8_4_8",
    ]:
        aie_vs_llvm_cpu(config, test_files_dir / f"{name}.mlir")

    for name in [
        "conv2d_nhwc_int32",
        "conv2d_nhwc_bf16",
        "conv2d_nhwc_int8",
        "conv2d_nhwc_q",
    ]:
        n_conv_repeats = 4

        aie_vs_llvm_cpu(
            config,
            test_files_dir / f"{name}.mlir",
            pipeline="conv-decompose",
            n_repeats=n_conv_repeats,
        )

    aie_vs_llvm_cpu(
        config,
        test_files_dir / "three_matmuls.mlir",
        function_name="three_$mm$",
    )

    matmul_template_dir = file_dir / "matmul_template"

    # Test(s) of the form matmul(A,B) where A:MxK, B:KxN
    test_name = output_dir / "test_from_template.mlir"
    template_name = matmul_template_dir / "matmul_MxK_KxN.mlir"
    generate_matmul_test(test_name, template_name, 32, 32, 64, "bf16", "f32")
    aie_vs_llvm_cpu(config, test_name)

    # Test(s) of the form matmul(A,B) + C where A:MxK, B:KxN, C:N
    test_name = output_dir / "test_from_template_bias_N.mlir"
    template_name = matmul_template_dir / "matmul_bias_MxK_KxN_N.mlir"
    generate_matmul_test(test_name, template_name, 1024, 1024, 512, "bf16", "f32")
    aie_vs_llvm_cpu(config, test_name, pipeline="pack-peel", use_ukernel=True)
    aie_vs_llvm_cpu(config, test_name, pipeline="pack-peel", use_ukernel=False)

    # Test(s) of the form matmul(A,B) + C where A:MxK, B:KxN, C:MxN
    test_name = output_dir / "test_from_template_full_bias.mlir"
    template_name = matmul_template_dir / "matmul_bias_MxK_KxN_MxN.mlir"
    generate_matmul_test(test_name, template_name, 128, 128, 256, "i32", "i32")
    aie_vs_llvm_cpu(config, test_name, pipeline="pack-peel", rtol=0, atol=0)

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
        prog="CPU comparison test",
        description="This program compares numerical outputs on AIE against CPU",
    )
    abs_path = lambda x: Path(x).absolute()
    parser.add_argument("output_dir", type=abs_path)
    parser.add_argument("iree_install_dir", type=abs_path)
    parser.add_argument(
        "peano_install_dir", nargs="?", default="/opt/llvm-aie", type=abs_path
    )
    parser.add_argument("xrt_dir", nargs="?", default="/opt/xilinx/xrt", type=abs_path)
    parser.add_argument(
        "vitis_dir", nargs="?", default="/opt/Xilinx/Vitis/2024.1", type=abs_path
    )

    # This (and other boolean flags) could be made more 'slick' by using
    # `action='store_true'` in the `add_argument` call, but this has
    # problems with the default value of 1. It could be also be made nicer
    # by using type=bool, but this also has issues. So going with this
    # clunky design for now (feel free to improve).

    cast_to_bool = lambda x: bool(x)
    parser.add_argument(
        "--return_on_fail",
        nargs="?",
        default=1,
        type=cast_to_bool,
        help=(
            "If 0, then the script will continue running even if a test fails, "
            "enumerating all failures. Otherwise the script will exit on the first failure."
        ),
    )

    parser.add_argument(
        "--verbose",
        nargs="?",
        default=1,
        type=cast_to_bool,
        help="If 0, then print statements are suppressed, otherwise they are printed.",
    )

    parser.add_argument(
        "--reset_npu_between_runs",
        nargs="?",
        default=1,
        type=cast_to_bool,
        help=(
            "If 0 then the NPU is not reset between runs, otherwise it is reset. "
            "Resetting between runs can in theory help avoid certain types of "
            "errors in parts of the stack which these tests are not designed to catch."
        ),
    )

    args = parser.parse_args()
    all_tests(
        args.output_dir,
        args.iree_install_dir,
        args.peano_install_dir,
        args.xrt_dir,
        args.vitis_dir,
        args.return_on_fail,
        args.verbose,
        args.reset_npu_between_runs,
    )
