# Copyright 2024 The IREE Authors

import sys
import numpy as np
import subprocess
import os
import urllib.request
import sys
from input_generator import generate_inputs, verify_determinism
from output_comparer import compare

sys.path.append(os.path.join(os.path.dirname(__file__), "matmul_template"))
from matmul_generator import generate_matmul_test


def find_executable(install_dir, executable_name):
    """
    Search for an executable in the given directory and its subdirectories
    'bin' and 'tools'. If the executable is not found, raise a RuntimeError.
    """
    search_dirs = [
        install_dir,
        os.path.join(install_dir, "bin"),
        os.path.join(install_dir, "tools"),
    ]
    for directory in search_dirs:
        executable_path = os.path.join(directory, executable_name)
        if os.path.isfile(executable_path):
            return executable_path
    raise RuntimeError(
        f"No '{executable_name}' executable found in '{install_dir}' or subdirectories."
    )


def assert_num_args(argv):
    if len(argv) < 2 or len(argv) > 5:
        error_message = (
            f"\n Illegal number of parameters: {len(argv)}, expected 2-5 parameters."
            "\n The parameters are as follows:"
            "\n     1) <output-dir>               (required)"
            "\n     2) <iree-install-dir>         (required)"
            "\n     3) <peano-install-dir>        (optional)"
            "\n     4) <xrt-dir>                  (optional)"
            "\n     5) <vitis-install-dir>        (optional)"
            "\n Possible example (dependent on environment variables):"
            "\n     python3 ./run_test.py "
            "results_dir_tmp  $IREE_INSTALL_DIR  "
            "$PEANO_INSTALL_DIR  /opt/xilinx/xrt  $VITIS_INSTALL_PATH"
        )
        raise RuntimeError(error_message)


def validate_directory(directory):
    if not os.path.isdir(directory):
        raise RuntimeError(f"'{directory}' is not a directory.")


def validate_file(file):
    if not os.path.isfile(file):
        raise RuntimeError(f"'{file}' is not a file.")


def arg_or_default(argv, index, default):
    path = default
    if len(argv) > index:
        path = argv[index]
    path = os.path.realpath(path)
    validate_directory(path)
    return path


def run_script(script, verbose):
    if verbose:
        print(f"Running the following script:\n{script}")

    bash_path = "/bin/bash"

    # Confirm that /bin/bash is available:
    if not os.path.isfile(bash_path):
        raise RuntimeError("The bash shell is required to run a test script.")

    process = subprocess.Popen(
        script,
        shell=True,
        executable=bash_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    if verbose:
        print("\nStandard output from script:")
        print(stdout.decode())
        print("\nStandard error from script:")
        print(stderr.decode())
        print("\n")
    if process.returncode != 0:
        raise RuntimeError(
            f"Error executing script, error code was {process.returncode}"
        )


def generate_aie_output(
    config,
    name,
    pipeline,
    use_ukernel,
    test_file,
    input_flags,
    function_name,
):
    """
    Compile and run a test file for AIE, returning a numpy array of the output.
    """

    peano_dir = config.peano_dir
    iree_install_dir = config.iree_install_dir
    vitis_dir = config.vitis_dir
    xrt_dir = config.xrt_dir
    iree_compile_exe = config.iree_compile_exe
    iree_run_exe = config.iree_run_exe
    output_dir = config.output_dir
    verbose = config.verbose

    aie_vmfb = os.path.join(output_dir, f"{name}_aie.vmfb")
    aie_npy = os.path.join(output_dir, f"{name}_aie.npy")

    compilation_flags = f"--iree-hal-target-backends=amd-aie \
 --iree-amdaie-tile-pipeline={pipeline} \
 --iree-amdaie-matmul-elementwise-fusion \
 --iree-amd-aie-peano-install-dir={peano_dir} \
 --iree-amd-aie-install-dir={iree_install_dir} \
 --iree-amd-aie-vitis-install-dir={vitis_dir} \
 --iree-hal-dump-executable-files-to={output_dir} \
 --iree-amd-aie-show-invoked-commands \
 --iree-scheduling-optimize-bindings=false \
 --mlir-disable-threading -o {aie_vmfb}"

    if use_ukernel:

        MM_KERNEL_URL = (
            "https://github.com/nod-ai/iree-amd-aie/releases/download/ukernels/mm.o"
        )
        compilation_flags += " --iree-amdaie-enable-ukernels=all"
        mm_fn = os.path.join(output_dir, "mm.o")
        if os.path.isfile(mm_fn):
            if verbose:
                print(f"File {mm_fn} already exists")
        else:
            if verbose:
                print(f"Attempting to download {MM_KERNEL_URL} to {mm_fn}")

            urllib.request.urlretrieve(MM_KERNEL_URL, mm_fn)

    function_line = ""
    if function_name:
        function_line = f"--function={function_name}"

    # We need to drop out of python to run the IREE compile command
    github_actions = "${GITHUB_ACTIONS}"
    reset_npu_script = config.reset_npu_script

    script = f"""
  # Fail on any error and print commands being run:
  set -ex

  # Source XRT setup script
  source {xrt_dir}/setup.sh

  # Get ready
  cd {output_dir}

  # Reset NPU (only in CI to facilitate testing without sudo access).
  if [ "{github_actions}" = true ]; then
    bash {reset_npu_script}
  fi

  # Compile
  {iree_compile_exe} {test_file} {compilation_flags}

  # Run
  {iree_run_exe} --module={aie_vmfb} {input_flags} \
          --device=xrt --output=@{aie_npy} {function_line}
    """

    # Replace all '$' symbols with '\$' to avoid bash expansion
    script = script.replace("$", "\\$")
    run_script(script, config.verbose)
    return np.load(aie_npy)


def generate_llvm_cpu_output(
    config,
    name,
    test_file,
    input_flags,
    function_name,
):
    """
    Compile and run a test file for IREE's CPU backend, returning a numpy array
    of the output.
    """
    iree_compile_exe = config.iree_compile_exe
    iree_run_exe = config.iree_run_exe
    output_dir = config.output_dir
    xrt_dir = config.xrt_dir

    cpu_vmfb = os.path.join(output_dir, f"{name}_cpu.vmfb")
    cpu_npy = os.path.join(output_dir, f"{name}_cpu.npy")
    compilation_flags = f"--iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-cpu-features=host -o {cpu_vmfb}"

    function_line = ""
    if function_name:
        function_line = f"--function={function_name}"

    script = f"""
  # Fail on any error and print commands being run:
  set -ex
  cd {output_dir}
 
  # Source XRT setup script
  source {xrt_dir}/setup.sh
 
  # Compile
  {iree_compile_exe} {test_file} {compilation_flags}
 
  # Run
  {iree_run_exe} --module={cpu_vmfb} {input_flags} \
          --output=@{cpu_npy} {function_line}
    """

    # Replace all '$' symbols with '\$' to avoid bash expansion
    script = script.replace("$", "\\$")
    run_script(script, config.verbose)
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
        file_parent_dir = os.path.dirname(file_dir)
        self.reset_npu_script = os.path.join(file_parent_dir, "reset_npu.sh")

        # Populated at runtime
        self.failures = []

        if not isinstance(self.verbose, bool) and not isinstance(self.verbose, int):
            raise ValueError(
                f"verbose must be a boolean or integer, not {type(verbose)}"
            )

    def __str__(self):
        return f"""
Settings used in all tests
==========================
xrt_dir:          {self.xrt_dir}
verbose:          {self.verbose}
file_dir:         {self.file_dir}
peano_dir:        {self.peano_dir}
vitis_dir:        {self.vitis_dir}
output_dir:       {self.output_dir}
iree_run_exe:     {self.iree_run_exe}
return_on_fail:   {self.return_on_fail}
iree_install_dir: {self.iree_install_dir}
reset_npu_script: {self.reset_npu_script}
iree_compile_exe: {self.iree_compile_exe}
        """

    def __repr__(self):
        return self.__str__()


def run_test(
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
    if n_repeats == 0:
        return

    name = os.path.basename(test_file).replace(".mlir", "")

    input_flags = generate_inputs(test_file, config.output_dir, seed)

    cpu_output = generate_llvm_cpu_output(
        config, "matmul_int32", test_file, input_flags, function_name
    )

    for i in range(n_repeats):
        if config.verbose:
            print(f"Run #{i+1} of {n_repeats} for {test_file}")

        aie_output = generate_aie_output(
            config,
            name,
            pipeline,
            use_ukernel,
            test_file,
            input_flags,
            function_name,
        )

        same_result = compare(cpu_output, aie_output, rtol, atol)
        if not same_result:
            config.failures.append(test_file)
            if config.return_on_fail:
                raise RuntimeError("Test failed, exiting.")


def run_all(argv):
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

    assert_num_args(argv)

    output_dir = os.path.realpath(argv[0])
    os.makedirs(output_dir, exist_ok=True)
    validate_directory(output_dir)

    iree_install_dir = os.path.realpath(argv[1])
    validate_directory(iree_install_dir)
    iree_compile_exe = find_executable(iree_install_dir, "iree-compile")
    iree_run_exe = find_executable(iree_install_dir, "iree-run-module")

    peano_dir = arg_or_default(argv, 2, "/opt/llvm-aie")
    xrt_dir = arg_or_default(argv, 3, "/opt/xilinx/xrt")
    vitis_dir = arg_or_default(argv, 4, "/opt/Xilinx/Vitis/2024.1")

    file_dir = os.path.dirname(os.path.realpath(__file__))

    # TODO(newling) expose these to user:
    verbose = True
    return_on_fail = True

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
    )
    if verbose:
        print(config)

    # Sanity check that results are reproducible across platforms
    verify_determinism()

    # Verify a very basic script runs before running the more complex tests
    test_script = "pwd"
    run_script(test_script, config.verbose)

    test_files_dir = os.path.join(file_dir, "test_files")
    for name in [
        "matmul_int32",
        "two_matmul_switching",
        "matmul_f32_8_8_4",
        "matmul_f32_8_4_8",
    ]:
        run_test(config, os.path.join(test_files_dir, f"{name}.mlir"))

    for name in [
        "conv2d_nhwc_int32",
        "conv2d_nhwc_bf16",
        "conv2d_nhwc_int8",
        "conv2d_nhwc_q",
    ]:
        n_conv_repeats = 4

        run_test(
            config,
            os.path.join(test_files_dir, f"{name}.mlir"),
            pipeline="conv-decompose",
            n_repeats=n_conv_repeats,
        )

    run_test(
        config,
        os.path.join(test_files_dir, "three_matmuls.mlir"),
        function_name="three_$mm$",
    )

    matmul_template_dir = os.path.join(file_dir, "matmul_template")

    # Test(s) of the form matmul(A,B) where A:MxK, B:KxN
    test_name = os.path.join(output_dir, "test_from_template.mlir")
    template_name = os.path.join(matmul_template_dir, "matmul_MxK_KxN.mlir")
    generate_matmul_test(test_name, template_name, 32, 32, 64, "bf16", "f32")
    run_test(config, test_name)

    # Test(s) of the form matmul(A,B) + C where A:MxK, B:KxN, C:N
    test_name = os.path.join(output_dir, "test_from_template_bias_N.mlir")
    template_name = os.path.join(matmul_template_dir, "matmul_bias_MxK_KxN_N.mlir")
    generate_matmul_test(test_name, template_name, 1024, 1024, 512, "bf16", "f32")
    run_test(config, test_name, pipeline="pack-peel", use_ukernel=True)
    run_test(config, test_name, pipeline="pack-peel", use_ukernel=False)

    # Test(s) of the form matmul(A,B) + C where A:MxK, B:KxN, C:MxN
    test_name = os.path.join(output_dir, "test_from_template_full_bias.mlir")
    template_name = os.path.join(matmul_template_dir, "matmul_bias_MxK_KxN_MxN.mlir")
    generate_matmul_test(test_name, template_name, 128, 128, 256, "i32", "i32")
    run_test(config, test_name, pipeline="pack-peel", rtol=0, atol=0)

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
    run_all(sys.argv[1::])
