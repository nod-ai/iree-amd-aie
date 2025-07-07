#!/usr/bin/env python3

# Copyright 2024 The IREE Authors

import argparse
import copy
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

import numpy as np

from convolution_template.convolution_generator import ConvolutionMlirGenerator
from input_generator import (
    generate_inputs,
    get_output_type,
    np_from_binfile,
    verify_determinism,
)
from matmul_template.matmul_generator import generate_matmul_test
from matmul_test_config import matmul_tests_for_each_device
from output_comparer import compare
from softmax_template.softmax_generator import generate_softmax_test


@dataclass
class TestParams(ABC):
    run_on_target: List[str] = field(default_factory=lambda: ["npu1_4col"])
    aie_compilation_flags: List[str] = field(default_factory=list)
    tile_pipeline: str = "pack-peel"
    lower_to_aie_pipeline: str = "objectFifo"
    name_suffix: str = ""
    use_chess: bool = False
    use_chess_for_ukernel: bool = True
    use_ukernel: bool = False
    run_benchmark: bool = False
    n_repeats: int = 1
    n_kernel_runs: int = 1
    n_reconfigure_runs: int = 1
    n_pdi_loads: int = 1
    enable_ctrlpkt: bool = True
    stack_size: int = 1024
    rtol: float = 1e-6
    atol: float = 1e-6
    preset_inputs: Dict[str, Any] = field(default_factory=dict)
    preset_output: Optional[Any] = None


class BaseTest(ABC):
    """
    Base class to be inherited by all tests.

    Args:
       name: The test name used to distinguish each other.

       function_name: The function name used in the MLIR test file.

       test_params: A collection of test parameters that are common to all tests.
    """

    def __init__(
        self,
        name="",
        function_name="",
        test_params=None,
    ):
        test_params = test_params if test_params is not None else TestParams()
        self.run_on_target = test_params.run_on_target
        self.aie_compilation_flags = test_params.aie_compilation_flags
        assert isinstance(self.aie_compilation_flags, list)
        assert all(isinstance(flag, str) for flag in self.aie_compilation_flags)

        # NB: derived classes should add labels to this list in their
        # constructor, never overwrite it.
        self.labels = ["All"]

        # Copy the test parameters to the instance.
        self.__dict__.update(test_params.__dict__)

        # Form test name.
        self.name = f"{name}_{self.name_suffix}" if self.name_suffix else name
        self.function_name = function_name

        if self.tile_pipeline == "pack-peel-4-level-tiling":
            self.name += "_4_level_tiling"

        if self.use_chess or self.use_chess_for_ukernel:
            self.labels.append("Chess")
        else:
            self.labels.append("Peano")

        if self.use_chess:
            self.name += f"_chess"
            self.add_aie_compilation_flags([f"--iree-amd-aie-enable-chess=1"])

        if self.use_ukernel:
            self.labels.append("UKernel")
            if self.use_chess_for_ukernel:
                self.name += "_ukernel_chess"
                self.add_aie_compilation_flags(
                    [f"--iree-amd-aie-enable-chess-for-ukernel=1"]
                )
            else:
                self.name += "_ukernel_peano"
                self.add_aie_compilation_flags(
                    [f"--iree-amd-aie-enable-chess-for-ukernel=0"]
                )

        if self.enable_ctrlpkt:
            self.labels.append("CtrlPkt")
            self.add_aie_compilation_flags(["--iree-amdaie-enable-control-packet=true"])
            # Ensure the packet flow flag is also set; add as `auto` if missing.
            if not any(
                flag.startswith("--iree-amdaie-packet-flow-strategy=")
                for flag in self.aie_compilation_flags
            ):
                self.add_aie_compilation_flags(
                    ["--iree-amdaie-packet-flow-strategy=auto"]
                )
            self.name += "_ctrlpkt"

        if self.run_benchmark:
            self.add_aie_compilation_flags(
                ["--iree-amdaie-enable-infinite-loop-around-core-block=true"]
            )
            self.labels.append("Performance")
            self.name += "_benchmark"
        else:
            self.labels.append("Correctness")

        self.add_aie_compilation_flags([f"--iree-amdaie-stack-size={self.stack_size}"])

    def add_aie_compilation_flags(self, flags):
        if flags:
            if isinstance(flags, str):
                flags = flags.split()
            assert isinstance(flags, list)
            assert all(isinstance(flag, str) for flag in flags)

            self.aie_compilation_flags += flags
            # unique-ify the list
            self.aie_compilation_flags = list(set(self.aie_compilation_flags))

    def run(self, config):
        # If the target device is not in the set of devices to run on, then
        # return False. ie. don't raise an error because is legitimate,
        # we just won't run the test.
        if config.target_device not in self.run_on_target:
            return False

        # If use_chess=1, and config has not provided a valid
        # path to vitis, then don't run the test. The asymmetry between
        # logic for peano and chess is because we don't expect everyone
        # running this script to have chess (currently Windows CI for example
        # does not).
        if self.use_chess and not config.vitis_dir:
            return False
        if self.use_ukernel and self.use_chess_for_ukernel and not config.vitis_dir:
            return False

        # If use_chess=0, and config has not provided a valid
        # path to peano, then bail: a path to peano must be provided.
        if not self.use_chess and not config.peano_dir:
            raise RuntimeError("Peano path not provided, and use_chess=False")
        if not self.use_chess_for_ukernel and not config.peano_dir:
            raise RuntimeError(
                "Peano path not provided, and use_chess_for_ukernel=False"
            )

        # Generate the MLIR file for the test.
        self.generate(config)

        # Choose to run either the benchmark test for measuring performance, or the vs_cpu test for correctness checking.
        if self.run_benchmark:
            return self.benchmark(config)
        else:
            return self.vs_cpu(config)

    @abstractmethod
    def generate(self, config):
        raise NotImplementedError("Subclasses must implement the generate method.")

    def get_dir(self, config):
        return config.get_test_dir(self.name)

    def get_filename(self, config):
        return self.get_dir(config) / f"{self.name}.mlir"

    def vs_cpu(self, config):
        filename = self.get_filename(config)

        aie_vs_llvm_cpu(
            config=config,
            aie_compilation_flags=self.aie_compilation_flags,
            test_file=filename,
            use_ukernel=self.use_ukernel,
            tile_pipeline=self.tile_pipeline,
            lower_to_aie_pipeline=self.lower_to_aie_pipeline,
            function_name=self.function_name,
            n_repeats=self.n_repeats,
            rtol=self.rtol,
            atol=self.atol,
            preset_inputs=self.preset_inputs,
            preset_output=self.preset_output,
        )
        return True

    def benchmark(self, config):
        filename = self.get_filename(config)

        benchmark_aie(
            config=config,
            aie_compilation_flags=self.aie_compilation_flags,
            test_file=filename,
            use_ukernel=self.use_ukernel,
            tile_pipeline=self.tile_pipeline,
            lower_to_aie_pipeline=self.lower_to_aie_pipeline,
            function_name=self.function_name,
            enable_ctrlpkt=self.enable_ctrlpkt,
            n_repeats=self.n_repeats,
            n_kernel_runs=self.n_kernel_runs,
            n_reconfigure_runs=self.n_reconfigure_runs,
            n_pdi_loads=self.n_pdi_loads,
        )

        return True


class ConvolutionFromTemplate(BaseTest):
    def __init__(
        self,
        generator,
        test_params=None,
    ):
        super().__init__(
            name=f"{generator.params['conv_type']}_{generator.params['N']}_{generator.params['IW']}_{generator.params['input_element_type']}_{generator.params['output_element_type']}",
            function_name="f_conv",
            test_params=test_params,
        )
        self.generator = generator
        # TODO(newling) Use all parameters in name, to avoid name collision.
        self.labels += ["Convolution"]

    def generate(self, config):
        filename = self.get_filename(config)
        self.generator.write_to_file(filename)


class MultipleDispatches(BaseTest):
    def __init__(
        self,
        file_base_name,
        function_name,
        test_params=None,
    ):
        super().__init__(
            name=file_base_name,
            function_name=function_name,
            test_params=test_params,
        )
        self.labels += ["Matmul", "MultipleDispatches"]
        self.file_base_name = file_base_name

    def generate(self, config):
        # Simply copy the test file to the working directory.
        test_filename = config.file_dir / "test_files" / f"{self.file_base_name}.mlir"
        shutil.copy2(test_filename, self.get_filename(config))


class BaseMatmul(BaseTest):
    def __init__(
        self,
        M,
        N,
        K,
        input_type,
        acc_type,
        name="",
        function_name="matmul",
        test_params=None,
    ):
        """
        Base class for all variants of dispatches with a matmul, currently
        matmuls, and matmuls with fused elementwise operations.
        """
        super().__init__(
            name=name,
            function_name=function_name,
            test_params=test_params,
        )
        self.labels.append("BaseMatmul")
        self.M = M
        self.N = N
        self.K = K
        self.input_type = input_type
        self.acc_type = acc_type

        self.labels.append(self.tile_pipeline)

        self.labels.append(self.lower_to_aie_pipeline)

        self.function_name = function_name

    def benchmark(self, config):
        status = super().benchmark(config)

        # Print some additional information that might have been tagged on earlier.
        if config.verbose:
            if hasattr(self, "n_matmul_ops"):
                print(f"Total ops (2 * M * N * K * repeats) : {self.n_matmul_ops}")

            if hasattr(self, "n_cols"):
                print(f"Number of columns : {self.n_cols}")

            if hasattr(self, "n_rows"):
                print(f"Number of rows : {self.n_rows}")

        return status

    def generate(self, config, template_name):
        generate_matmul_test(
            self.get_filename(config),
            template_name,
            self.M,
            self.N,
            self.K,
            self.input_type,
            self.acc_type,
        )


class Matmul(BaseMatmul):
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
        additional_labels=None,
        test_params=None,
    ):
        super().__init__(
            name=f"matmul_{M}_{N}_{K}_{input_type}_{acc_type}",
            test_params=test_params,
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
        )
        self.labels.append("Matmul")

        if additional_labels:
            self.labels += additional_labels
        if self.run_benchmark:
            self.labels.append("MatmulBenchmark")

    def generate(self, config):
        template_name = config.file_dir / "matmul_template" / "matmul_MxK_KxN.mlir"
        super().generate(config, template_name)


class MatmulTransposeB(BaseMatmul):
    """
    A test of the form matmul_transpose_b(A,B) where A:MxK, B:NxK
    """

    def __init__(
        self,
        M,
        N,
        K,
        input_type,
        acc_type,
        additional_labels=None,
        test_params=None,
    ):
        super().__init__(
            name=f"matmul_transpose_b_{M}_{N}_{K}_{input_type}_{acc_type}",
            test_params=test_params,
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
            function_name="matmul_transpose_b",
        )
        self.labels.append("MatmulTransposeB")

        if additional_labels:
            self.labels += additional_labels
        if self.run_benchmark:
            self.labels.append("MatmulTransposeBBenchmark")

    def generate(self, config):
        template_name = (
            config.file_dir / "matmul_template" / "matmul_transpose_b_MxK_NxK.mlir"
        )
        super().generate(config, template_name)


class MatmulTransposeA(BaseMatmul):
    """
    A test of the form matmul_transpose_a(A,B) where A:KxM, B:KxN
    """

    def __init__(
        self,
        M,
        N,
        K,
        input_type,
        acc_type,
        additional_labels=None,
        test_params=None,
    ):
        super().__init__(
            name=f"matmul_transpose_a_{M}_{N}_{K}_{input_type}_{acc_type}",
            test_params=test_params,
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
            function_name="matmul_transpose_a",
        )
        self.labels.append("MatmulTransposeA")

        if additional_labels:
            self.labels += additional_labels
        if self.run_benchmark:
            self.labels.append("MatmulTransposeABenchmark")

    def generate(self, config):
        template_name = (
            config.file_dir / "matmul_template" / "matmul_transpose_a_KxM_KxN.mlir"
        )
        super().generate(config, template_name)


class Matmul4d(BaseMatmul):
    """
    A test of linalg.generic with 4d inputs and output, following the form:
    C += matmul4d(A,B) where A:M1xK1xM0xK0, B:N1xK1xK0xN0, C:N1xM1xM0xN0

    -- M0/N0/K0 are inner dim sizes, currently fixed at 32/32/64 for comparison purpose.

    -- M1/N1/K1 are outer dim sizes.
       Note that the outer dims for this operation are transposed to make sure
       successful compilation through LogicalObjectFifo pipeline.

    -- The input parameters M/N/K are the total size which equals to the product
       of outer and inner dim sizes.
    """

    def __init__(
        self,
        M,
        N,
        K,
        input_type,
        acc_type,
        M0=32,
        N0=32,
        K0=64,
        additional_labels=None,
        test_params=None,
    ):
        super().__init__(
            name=f"matmul4d_{M}_{N}_{K}_{input_type}_{acc_type}",
            test_params=test_params,
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
            function_name="matmul4d",
        )
        self.M0 = M0
        self.N0 = N0
        self.K0 = K0
        self.labels.append("Matmul4d")
        if additional_labels:
            self.labels += additional_labels
        if self.run_benchmark:
            self.labels.append("Matmul4dBenchmark")

    def generate(self, config):
        template_name = (
            config.file_dir
            / "matmul_template"
            / "matmul4d_M1xK1xM0xK0_N1xK1xK0xN0.mlir"
        )

        generate_matmul_test(
            self.get_filename(config),
            template_name,
            self.M,
            self.N,
            self.K,
            self.input_type,
            self.acc_type,
            m0=self.M0,
            n0=self.N0,
            k0=self.K0,
        )


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
        test_params=None,
    ):
        super().__init__(
            name=f"matmul_thin_bias_{M}_{N}_{K}_{input_type}_{acc_type}",
            function_name="matmul_bias",
            test_params=(
                test_params
                if test_params is not None
                else TestParams(lower_to_aie_pipeline="air", enable_ctrlpkt=False)
            ),
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
        )
        self.labels.append("MatmulThinBias")

        self.add_aie_compilation_flags(
            [
                "--iree-amdaie-matmul-elementwise-fusion",
                "--iree-amdaie-num-rows=2",
                "--iree-amdaie-num-cols=2",
            ]
        )

    def generate(self, config):
        template_name = (
            config.file_dir / "matmul_template" / "matmul_bias_MxK_KxN_N.mlir"
        )
        super().generate(config, template_name)


class MatmulFullBias(BaseMatmul):
    """
    A test of the form matmul(A,B) + C where A:MxK, B:KxN, C:MxN
    """

    def __init__(
        self,
        M,
        N,
        K,
        input_type,
        acc_type,
        test_params=None,
    ):
        super().__init__(
            name=f"matmul_full_bias_{M}_{N}_{K}_{input_type}_{acc_type}",
            function_name="matmul_bias",
            test_params=(
                test_params
                if test_params is not None
                else TestParams(lower_to_aie_pipeline="air", enable_ctrlpkt=False)
            ),
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
        )
        self.labels.append("MatmulFullBias")

        self.add_aie_compilation_flags(
            [
                "--iree-amdaie-matmul-elementwise-fusion",
                "--iree-amdaie-num-rows=2",
                "--iree-amdaie-num-cols=2",
            ]
        )

    def generate(self, config):
        template_name = (
            config.file_dir / "matmul_template" / "matmul_bias_MxK_KxN_MxN.mlir"
        )
        super().generate(config, template_name)


class BatchMatmul(BaseMatmul):
    """
    A test of the form batch_matmul(A,B) where A:BxMxK, B:BxKxN
    """

    def __init__(
        self,
        B,
        M,
        N,
        K,
        input_type,
        acc_type,
        test_params=None,
    ):
        super().__init__(
            name=f"batch_matmul_{B}_{M}_{N}_{K}_{input_type}_{acc_type}",
            function_name="batch_matmul",
            test_params=test_params,
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
        )
        self.labels.append("BatchMatmul")
        self.B = B

    def generate(self, config):
        template_name = (
            config.file_dir / "matmul_template" / "batch_matmul_BxMxK_BxKxN.mlir"
        )

        generate_matmul_test(
            self.get_filename(config),
            template_name,
            self.M,
            self.N,
            self.K,
            self.input_type,
            self.acc_type,
            b=self.B,
        )


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
        test_params=None,
    ):
        super().__init__(
            name=f"matmul_truncf_{M}_{K}_{input_type}_{acc_type}",
            function_name="matmul_truncf",
            test_params=test_params,
            M=M,
            N=M,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
        )
        self.labels.append("MatmulTruncf")

    def generate(self, config):
        template_name = (
            config.file_dir / "matmul_template" / "matmul_truncf_MxK_KxN.mlir"
        )
        super().generate(config, template_name)


class MatmulScaleTrunci(BaseMatmul):
    """
    A test of the form matmul(A,B) + scale(C) + trunci(C) where A:MxK, B:KxN and C:MxN
    """

    def __init__(
        self,
        M,
        N,
        K,
        input_type,
        acc_type,
        test_params=None,
    ):
        super().__init__(
            name=f"matmul_scale_trunci_{M}_{N}_{K}_{input_type}_{acc_type}",
            function_name="matmul_trunci",
            test_params=test_params,
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
        )
        self.labels.append("MatmulScaleTrunci")

    def generate(self, config):
        template_name = (
            config.file_dir / "matmul_template" / "matmul_trunci_scaling_MxK_KxN.mlir"
        )
        super().generate(config, template_name)


class Matmul4dScaleTrunci(BaseMatmul):
    """
    A test of the form C = matmul4d(A,B) + scale(C) + trunci(C)
    where A:M1xK1xM0xK0, B:N1xK1xK0xN0, C:N1xM1xM0xN0
    """

    def __init__(
        self,
        M,
        N,
        K,
        input_type,
        acc_type,
        M0=64,
        N0=64,
        K0=128,
        additional_labels=None,
        test_params=None,
    ):
        super().__init__(
            name=f"matmul4d_scale_trunci_{M}_{N}_{K}_{input_type}_{acc_type}",
            test_params=test_params,
            M=M,
            N=N,
            K=K,
            input_type=input_type,
            acc_type=acc_type,
            function_name="matmul4d_trunci",
        )
        self.M0 = M0
        self.N0 = N0
        self.K0 = K0
        self.labels.append("Matmul4dScaleTrunci")
        if additional_labels:
            self.labels += additional_labels
        if self.run_benchmark:
            self.labels.append("Matmul4dScaleTrunciBenchmark")

    def generate(self, config):
        template_name = (
            config.file_dir / "matmul_template" / "matmul4d_trunci_scaling.mlir"
        )
        generate_matmul_test(
            self.get_filename(config),
            template_name,
            self.M,
            self.N,
            self.K,
            self.input_type,
            self.acc_type,
            m0=self.M0,
            n0=self.N0,
            k0=self.K0,
        )


class Softmax(BaseTest):
    def __init__(
        self,
        M,
        N,
        data_type,
        test_params=None,
    ):
        super().__init__(
            name="softmax_{}_{}_{}".format(M, N, data_type),
            function_name="softmax",
            test_params=test_params,
        )
        self.labels += ["Softmax"]

        self.M = M
        self.N = N
        self.data_type = data_type

        if self.run_benchmark:
            self.labels.append("SoftmaxBenchmark")

    def generate(self, config):
        template_name = config.file_dir / "softmax_template" / "softmax_MxN.mlir"
        generate_softmax_test(
            self.get_filename(config),
            template_name,
            m=self.M,
            n=self.N,
            data_type=self.data_type,
        )


class Reduction(BaseTest):
    def __init__(
        self,
        file_base_name,
        function_name,
        test_params=None,
    ):
        super().__init__(
            name=file_base_name,
            test_params=test_params,
        )
        self.labels += ["Reduction"]
        self.file_base_name = file_base_name
        self.function_name = function_name

    def generate(self, config):
        # Simply copy the test file to the working directory.
        test_filename = config.file_dir / "test_files" / f"{self.file_base_name}.mlir"
        shutil.copy2(test_filename, self.get_filename(config))


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


def print_program_memory_size(test_dir):
    # Get all the .elf files in `test_dir`.
    # These elfs contain many sections, one of which is the program memory. Some digging into the elf format
    # see https://github.com/newling/aie-rt/commit/d0f08bc4a37092a919d6a0d51a44d9f0ae274bb9
    # revealed that the size of the program memory is stored at byte 72 of the elf.
    # This might change in the future, but for now this works reliably.
    #
    # Note that if the elfs are created with chess, then there are 2 sections of program memory, the second
    # is at byte 108. For now we ignore this case, this function should not be called if the
    # elfs are created with chess.
    elfs = list(test_dir.glob("*.elf"))
    number_of_elfs = len(elfs)

    if number_of_elfs == 0:
        print(
            f"There are no .elf files in {test_dir}, cannot determine program memory size"
        )
        return

    magic_byte = 72

    max_pm_size = 0
    for elf_file in elfs:
        with open(elf_file, "rb") as f:
            elf = f.read()
            pm = int.from_bytes(elf[magic_byte : magic_byte + 4], "little")
            max_pm_size = max(max_pm_size, pm)

    # Sanity check on the magic byte. If this really is the program memory,
    # it should not exceed 16384 bytes here.
    if max_pm_size > 16384:
        raise RuntimeError(
            f"Program memory size determined to be {max_pm_size} bytes, which is too large. This is likely not the program memory size, because if it were then an error would have been raised earlier."
        )
    if max_pm_size < 100:
        raise RuntimeError(
            f"Program memory size determined to be {max_pm_size} bytes, which is too small. This is likely not the program memory size and the 'magic byte' approach isn't valid."
        )

    print(f"There are {number_of_elfs} .elf file(s) in {test_dir}")
    print(
        f"The largest program memory size (read from byte {magic_byte} of elf files) is {max_pm_size} bytes"
    )


def generate_aie_vmfb(
    config,
    aie_compilation_flags,
    name,
    tile_pipeline,
    lower_to_aie_pipeline,
    use_ukernel,
    test_file,
):
    """
    Compile a test file for IREE's AIE backend, returning the path to the
    compiled module.
    """

    additional_flags = aie_compilation_flags

    test_dir = config.get_test_dir(name)

    aie_compilation_flags = [
        config.iree_compile_exe,
        test_file,
        "--iree-hal-target-backends=amd-aie",
        f"--iree-amdaie-target-device={config.target_device}",
        f"--iree-amdaie-tile-pipeline={tile_pipeline}",
        f"--iree-amdaie-lower-to-aie-pipeline={lower_to_aie_pipeline}",
        f"--iree-amd-aie-peano-install-dir={config.peano_dir}",
        f"--iree-amd-aie-install-dir={config.iree_dir}",
        f"--iree-amd-aie-vitis-install-dir={config.vitis_dir}",
        f"--iree-hal-dump-executable-files-to={test_dir}",
        f"--iree-amdaie-device-hal={config.device_hal}",
        "--iree-scheduling-optimize-bindings=false",
        "--iree-hal-memoization=false",
        "--iree-hal-indirect-command-buffers=false",
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
        test_dir / f"{name}_aie.vmfb",
    ]

    start = time.monotonic_ns()
    shell_out(aie_compilation_flags, test_dir, config.verbose)
    compile_time = time.monotonic_ns() - start
    if config.verbose:
        print(f"Time spent in compilation: {compile_time // 1e6} [ms]")

    aie_vmfb = test_dir / f"{name}_aie.vmfb"
    if not aie_vmfb.exists():
        raise RuntimeError(f"Failed to compile {test_file} to {aie_vmfb}")

    return aie_vmfb


def generate_aie_output(config, aie_vmfb, input_args, function_name, name, output_type):
    """
    Run a compiled AIE module (aie_vmfb), returning a numpy array of the output.
    """

    test_dir = config.get_test_dir(name)
    aie_bin = test_dir / f"{name}_aie.bin"
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
    shell_out(run_args, test_dir, config.verbose)
    run_time = time.monotonic_ns() - start

    if config.verbose:
        print(f"Time spent in running the model: {run_time // 1e6} [ms]")

    return np_from_binfile(aie_bin, output_type)


def validate_run_params(
    n_kernel_runs, n_reconfigure_runs, n_pdi_loads, enable_ctrlpkt, batch_size=None
):
    # Valid Examples:
    # - n_kernel_runs=0, n_reconfigure_runs=0, n_pdi_loads=10 (measure PDI load time)
    # - n_kernel_runs=10, n_reconfigure_runs=0, n_pdi_loads=1 (measure kernel time, control packet reconfiguration disabled)
    # - n_kernel_runs=10, n_reconfigure_runs=1, n_pdi_loads=1 (measure kernel time, control packet reconfiguration enabled)
    # - n_kernel_runs=0, n_reconfigure_runs=10, n_pdi_loads=1 (measure reconfigure time)
    # - n_kernel_runs=10, n_reconfigure_runs=10, n_pdi_loads=1 (measure kernel and reconfigure time)

    # Check minimum values.
    min_kernel_runs = 0
    # When control packet enabled, reconfiguration run needs to be at least 1 for any subsequent kernel to work.
    min_reconfigure_runs = 1 if (n_kernel_runs > 0 and enable_ctrlpkt) else 0
    # PDI needs to be loaded at least once for any subsequent reconfiguration/kernel to work.
    min_pdi_loads = 1
    if n_kernel_runs < min_kernel_runs:
        raise ValueError(
            f"n_kernel_runs must be >= {min_kernel_runs}, got {n_kernel_runs}"
        )
    if n_reconfigure_runs < min_reconfigure_runs:
        raise ValueError(
            f"n_reconfigure_runs must be >= {min_reconfigure_runs}, got {n_reconfigure_runs}"
        )
    if n_pdi_loads < min_pdi_loads:
        raise ValueError(f"n_pdi_loads must be >= {min_pdi_loads}, got {n_pdi_loads}")

    # When batch size is provided, we are doing performance benchmarking.
    # Check consistency between run values and batch size so that performance results can be averaged correctly.
    if batch_size is not None:
        if n_kernel_runs > min_kernel_runs and n_kernel_runs != batch_size:
            raise ValueError(
                f"n_kernel_runs={n_kernel_runs} must match batch size {batch_size}."
            )
        if (
            n_reconfigure_runs > min_reconfigure_runs
            and n_reconfigure_runs != batch_size
        ):
            raise ValueError(
                f"n_reconfigure_runs={n_reconfigure_runs} must match batch size {batch_size}."
            )
        if n_pdi_loads > min_pdi_loads and n_pdi_loads != batch_size:
            raise ValueError(
                f"n_pdi_loads={n_pdi_loads} must match batch size {batch_size} when > 1."
            )


def benchmark_aie_kernel_time(
    config,
    aie_vmfb,
    input_args,
    function_name,
    name,
    enable_ctrlpkt,
    n_repeats,
    n_kernel_runs,
    n_reconfigure_runs,
    n_pdi_loads,
    time_unit,
):
    """
    Benchmark a compiled AIE module's (aie_vmfb) kernel time, average over the specified number of runs.
    """
    test_dir = config.get_test_dir(name)

    batch_size = max(n_kernel_runs, n_reconfigure_runs, n_pdi_loads)
    validate_run_params(
        n_kernel_runs, n_reconfigure_runs, n_pdi_loads, enable_ctrlpkt, batch_size
    )

    run_args = [
        config.iree_benchmark_exe,
        f"--module={aie_vmfb}",
        *input_args,
        f"--device={config.device_hal}",
        f"--benchmark_repetitions={n_repeats}",
        f"--batch_size={batch_size}",
        f"--xrt_lite_n_kernel_runs={n_kernel_runs}",
        f"--xrt_lite_n_reconfigure_runs={n_reconfigure_runs}",
        f"--xrt_lite_n_pdi_loads={n_pdi_loads}",
        f"--time_unit={time_unit}",
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
    shell_out(run_args, test_dir, config.verbose)
    run_time = time.monotonic_ns() - start

    if config.verbose:
        print(f"Time spent in running the model: {run_time // 1e6} [ms]")
    return True


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

    test_dir = config.get_test_dir(name)
    cpu_vmfb = test_dir / f"{name}_cpu.vmfb"
    aie_compilation_flags = [
        config.iree_compile_exe,
        test_file,
        "--iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu-features=host",
        "-o",
        f"{cpu_vmfb}",
    ]
    shell_out(aie_compilation_flags, workdir=test_dir, verbose=config.verbose)

    cpu_bin = test_dir / f"{name}_cpu.bin"
    run_args = [
        config.iree_run_exe,
        f"--module={cpu_vmfb}",
        *input_args,
        f"--output=@{cpu_bin}",
    ]
    if function_name:
        run_args += [f"--function={function_name}"]
    shell_out(run_args, workdir=test_dir, verbose=config.verbose)
    return np_from_binfile(cpu_bin, output_type)


class TestConfig:
    """
    Global state used for all tests. Stores paths to executables used.
    """

    def get_test_dir(self, test_name):
        """
        Return the subdirectory `test_name` of `output_dir`.
        (1) Assert that `test_name` is a 'good' name (no '.' no whitespace, etc)
        (2) Assert that `output_dir` / `test_name` exists and is a directory (create if not)
        """
        if not test_name.isidentifier():
            raise ValueError(f"test_name '{test_name}' is not a valid identifier")
        test_dir = self.output_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        return test_dir

    def __init__(
        self,
        output_dir,
        iree_dir,
        peano_dir,
        xrt_dir,
        vitis_dir,
        file_dir,
        iree_benchmark_exe,
        iree_compile_exe,
        iree_run_exe,
        verbose,
        reset_npu_between_runs,
        do_not_run_aie,
        device_hal,
        xrt_lite_n_core_rows,
        xrt_lite_n_core_cols,
        target_device,
    ):
        self.output_dir = output_dir
        self.iree_dir = iree_dir
        self.peano_dir = peano_dir
        self.xrt_dir = xrt_dir
        self.vitis_dir = vitis_dir
        self.file_dir = file_dir
        self.iree_benchmark_exe = iree_benchmark_exe
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

    def __str__(self):
        return dedent(
            f"""
        Settings and versions used in all tests
        -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
        target_device:        {self.target_device}
        do_not_run_aie:       {self.do_not_run_aie}
        file_dir:             {self.file_dir}
        iree_compile_exe:     {self.iree_compile_exe}
        iree_dir:             {self.iree_dir}
        iree_run_exe:         {self.iree_run_exe}
        kernel_version:       {self.linux_kernel}
        output_dir:           {self.output_dir}
        peano_commit_hash:    {self.peano_commit_hash}
        peano_dir:            {self.peano_dir}
        reset_npu_script:     {self.reset_npu_script}
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
    rtol,
    atol,
    n_repeats,
    output_type,
):
    """
    Arguments to the function are:
    config:
        TestConfig containing any state which is common to all tests
    aie_compilation_flags:
        A list of additional compilation flags to be passed to the AIE backend
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
    rtol:
        The relative tolerance to use when comparing the output of the
        AIE backend to the baseline value.
    atol:
        The absolute tolerance to use when comparing the output of the
        AIE backend to the baseline value.
    n_repeats:
        The number of times to run the test. This is useful for tests which
        may pass only sometimes due to driver issues, etc.
    output_type:
        The type of the output to be expected from the AIE backend.
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

    if config.verbose:
        # Check if "enable-chess=1" is a substring of any of the compilation flags:
        uses_chess = any("enable-chess=1" in flag for flag in aie_compilation_flags)
        if not uses_chess:
            test_dir = config.get_test_dir(name)
            print_program_memory_size(test_dir)


def benchmark_aie(
    config,
    aie_compilation_flags,
    test_file,
    use_ukernel,
    tile_pipeline,
    lower_to_aie_pipeline,
    function_name,
    enable_ctrlpkt,
    n_repeats,
    n_kernel_runs,
    n_reconfigure_runs,
    n_pdi_loads,
    seed=1,
    time_unit="us",
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
    use_ukernel:
        Whether to use micro-kernels when running on the AIE backend
    tile_pipeline:
        The tiling pipeline to use when compiling for the AIE backend
    lower_to_aie_pipeline:
        The pipeline to be used for lowering to AIE (objectFifo, AIR).
    enable_ctrlpkt:
        Whether to enable control packet reconfiguration.
    n_repeats:
        The number of repetitions to be used for getting statistics (mean, median, stddev)
    n_kernel_runs:
        The number of invocations of the kernel, for averaging.
    n_reconfigure_runs:
        The number of reconfiguration invocations, for averaging.
    n_pdi_loads:
        The number of PDI loads, for averaging.
    function_name:
        The name of the function to run (the test file may contain multiple
        functions).
    seed:
        The seed to be used for generating the inputs.
    time_unit:
        The time unit to be shown in the benchmark output (ns, us, ms).
    """
    if (
        "--iree-amdaie-enable-infinite-loop-around-core-block=true"
        not in aie_compilation_flags
    ):
        raise ValueError(
            "To benchmark an AIE kernel module, the "
            "`--iree-amdaie-enable-infinite-loop-around-core-block=true` "
            "should be passed."
        )

    name = name_from_mlir_filename(test_file)
    test_dir = config.get_test_dir(name)
    input_args = generate_inputs(test_file, test_dir, seed)

    aie_vmfb = generate_aie_vmfb(
        config,
        aie_compilation_flags,
        name,
        tile_pipeline,
        lower_to_aie_pipeline,
        use_ukernel,
        test_file,
    )

    if config.do_not_run_aie:
        if config.verbose:
            print(f"Skipping AIE run for {test_file} because 'do_not_run_aie=True'.")
        return

    print(f"Performance benchmark: {test_file}")
    benchmark_aie_kernel_time(
        config,
        aie_vmfb,
        input_args,
        function_name,
        name,
        enable_ctrlpkt,
        n_repeats,
        n_kernel_runs,
        n_reconfigure_runs,
        n_pdi_loads,
        time_unit,
    )

    if config.verbose:
        # Check if "enable-chess=1" is a substring of any of the compilation flags:
        uses_chess = any("enable-chess=1" in flag for flag in aie_compilation_flags)
        if not uses_chess:
            print_program_memory_size(test_dir)


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
    preset_inputs={},
    preset_output=None,
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

    test_dir = config.get_test_dir(name)

    input_args = generate_inputs(test_file, test_dir, seed, preset_inputs)
    output_type = get_output_type(test_file)

    if preset_output is not None:
        # If a preset output is provided, use it as the baseline.
        cpu_output = preset_output
    else:
        # Otherwise, generate the output using the cpu backend.
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

        # Tests Matmul + Trunci with Scaling.
        matmul_scale_trunci_tests = [
            # Phoenix : Ukernel + Peano.
            {"run_on_target": ["npu1_4col"]},
            # Phoenix : Vectorization + Peano.
            {"run_on_target": ["npu1_4col"], "use_ukernel": True},
            # Strix : Ukernel + Peano.
            {
                "run_on_target": ["npu4"],
                "use_chess": False,
                "use_ukernel": True,
                "use_chess_for_ukernel": False,
            },
        ]
        for test in matmul_scale_trunci_tests:
            lhs = 2 * np.ones([256, 128], dtype=np.int8)
            rhs = 3 * np.ones([128, 256], dtype=np.int8)
            excepted_out = 60 * np.ones([256, 256], dtype=np.int8)
            test_params = TestParams(
                tile_pipeline="pack-peel-4-level-tiling",
                run_on_target=test["run_on_target"],
                stack_size=3072,
                preset_inputs={1: lhs, 2: rhs},
                preset_output=excepted_out,
                rtol=0,
                atol=0,
            )
            if "use_ukernel" in test:
                test_params.use_ukernel = test["use_ukernel"]
            if "use_chess" in test:
                test_params.use_chess = test["use_chess"]
            if "use_chess_for_ukernel" in test:
                test_params.use_chess_for_ukernel = test["use_chess_for_ukernel"]
            self.register(
                MatmulScaleTrunci(
                    256,
                    256,
                    128,
                    "i8",
                    "i32",
                    test_params=test_params,
                )
            )

        # Tests Matmul + Truncf
        matmul_truncf_tests = [
            {
                "M": 16,
                "K": 16,
                "lhs": 101 * np.ones([16, 16]),
                "rhs": 3 * np.eye(16),
                "expected_out": 302 * np.ones([16, 16]),
            },
            {
                "M": 128,
                "K": 256,
                "lhs": 2 * np.ones([128, 256]),
                "rhs": 3 * np.ones([256, 128]),
                "expected_out": 1536 * np.ones([128, 128]),
            },
        ]
        for tile_pipeline in ["pack-peel", "pack-peel-4-level-tiling"]:
            for test in matmul_truncf_tests:
                self.register(
                    MatmulTruncf(
                        test["M"],
                        test["K"],
                        "bf16",
                        "f32",
                        test_params=TestParams(
                            tile_pipeline=tile_pipeline,
                            preset_inputs={1: test["lhs"], 2: test["rhs"]},
                            preset_output=test["expected_out"],
                            rtol=0,
                            atol=0,
                        ),
                    )
                )

        # BatchMatmul test(s):
        # TODO(jornt): BatchMatmul tests with the pack-peel-4-level-tiling pipeline result in intermittent
        # numerics issues. Re-enable.
        for tile_pipeline in ["pack-peel"]:
            for input_type, acc_type in zip(["i32", "bf16"], ["i32", "f32"]):
                # Batch size = 1:
                self.register(
                    BatchMatmul(
                        1,
                        128,
                        128,
                        256,
                        input_type,
                        acc_type,
                        test_params=TestParams(tile_pipeline=tile_pipeline),
                    )
                )
                # Batch size = 2:
                self.register(
                    BatchMatmul(
                        2,
                        64,
                        64,
                        64,
                        input_type,
                        acc_type,
                        test_params=TestParams(tile_pipeline=tile_pipeline),
                    )
                )
        # Strix + pack-peel-4-level-tiling + 4x8 + i32->i32.
        # TODO(avarma): Currently bf16->f32 vectorization is not supported for npu4.
        #               Enable the same once it is.
        self.register(
            BatchMatmul(
                1,
                128,
                128,
                256,
                "i32",
                "i32",
                test_params=TestParams(
                    tile_pipeline="pack-peel-4-level-tiling",
                    run_on_target=["npu4"],
                    name_suffix="4x8_npu4",
                    n_repeats=10,
                ),
            )
        )
        # Batch size = 2:
        self.register(
            BatchMatmul(
                2,
                64,
                64,
                64,
                "i32",
                "i32",
                test_params=TestParams(
                    tile_pipeline="pack-peel-4-level-tiling",
                    run_on_target=["npu4"],
                    name_suffix="4x8_npu4",
                    n_repeats=10,
                ),
            )
        )

        # MatmulThinBias test(s):
        self.register(MatmulThinBias(1024, 1024, 512, "bf16", "f32"))

        # MatmulFullBias test:
        self.register(MatmulFullBias(128, 128, 256, "bf16", "f32"))

        # MatmulTransposeB test(s):
        for input_type, acc_type in zip(["i8", "bf16"], ["i32", "f32"]):
            self.register(MatmulTransposeB(32, 32, 32, input_type, acc_type))
            self.register(MatmulTransposeB(128, 256, 128, input_type, acc_type))
            self.register(
                MatmulTransposeB(
                    128,
                    256,
                    128,
                    input_type,
                    acc_type,
                    test_params=TestParams(
                        tile_pipeline="pack-peel-4-level-tiling",
                        name_suffix="4level",
                    ),
                )
            )
            self.register(MatmulTransposeB(1536, 1536, 2048, input_type, acc_type))

        # MatmulTransposeA test(s):
        # Note: i8->i32 tests don't work because of the following op
        # %10 = vector.transpose %9, [1, 0] : vector<8x4xi8> to vector<4x8xi8>
        # failed to lowering to an `aievec.shuffle` op.
        for input_type, acc_type in zip(["i32", "bf16"], ["i32", "f32"]):
            self.register(MatmulTransposeA(32, 32, 32, input_type, acc_type))
            self.register(MatmulTransposeA(128, 256, 128, input_type, acc_type))
            self.register(MatmulTransposeA(1536, 1536, 2048, input_type, acc_type))

        for device in matmul_tests_for_each_device:
            for test in matmul_tests_for_each_device[device]:
                test_params = TestParams(run_on_target=[device], stack_size=3072)
                if "use_ukernel" in test:
                    test_params.use_ukernel = test["use_ukernel"]
                if "use_chess" in test:
                    test_params.use_chess = test["use_chess"]
                if "use_chess_for_ukernel" in test:
                    test_params.use_chess_for_ukernel = test["use_chess_for_ukernel"]
                if "name_suffix" in test:
                    test_params.name_suffix = test["name_suffix"]
                if "tile_pipeline" in test:
                    test_params.tile_pipeline = test["tile_pipeline"]
                if "aie_compilation_flags" in test:
                    test_params.aie_compilation_flags = test["aie_compilation_flags"]
                if "enable_ctrlpkt" in test:
                    test_params.enable_ctrlpkt = test["enable_ctrlpkt"]
                additional_labels = (
                    test["additional_labels"] if "additional_labels" in test else None
                )
                self.register(
                    Matmul(
                        test["M"],
                        test["N"],
                        test["K"],
                        test["input_type"],
                        test["acc_type"],
                        test_params=test_params,
                        additional_labels=additional_labels,
                    )
                )

        performance_tests = []

        ##############
        # NPU1 Tests #
        ##############

        performance_repl_base_dict = {
            "M": 512,
            "N": 512,
            "K": 512,
            "additional_labels": ["CorePerformance"],
            "aie_compilation_flags": [
                "--iree-amdaie-num-rows=1",
                "--iree-amdaie-num-cols=1",
            ],
            # effectively this says:
            #   for 3 launches:
            #     for 1 copy of data to and from AIE:
            #       for 100 calls to the matmul function:
            #          compute
            "call_replication": 100,
            "n_performance_repeats": 3,
            "n_performance_kernel_runs": 1,
        }

        # Test performance of core code generated with peano.
        # A matmul of shape 512x512x512, run on a single core,  in a loop 100
        # times without any data copy to/from core memory. Peak performance for
        # the 4x4 phoenix array is 4 TFlops (at 1GHz), so peak performace for single core
        # is 4/16 = 0.25 TFlops. This matmul is 512x512x512*2 flops = 0.25 GFlops.
        # So at peak performance, to run this 100 times should take
        # 0.1 seconds (100'000 microseconds). Hawk point is clocked faster than
        # 1 GHz, so at peak, less than 0.1 seconds.

        for opt_level, target in [[2, "npu1_4col"], [3, "npu1_4col"]]:
            performance_dict = copy.deepcopy(performance_repl_base_dict)
            performance_dict["peano_opt_level"] = opt_level
            performance_dict["run_on_target"] = target
            performance_dict["use_ukernel"] = False
            performance_tests.append(performance_dict)

        for target, chess_for_ukernel, in_type in [
            ["npu1_4col", True, "bf16"],
            ["npu4", False, "i8"],
        ]:
            performance_dict = copy.deepcopy(performance_repl_base_dict)
            performance_dict["in_dtype"] = in_type
            performance_dict["run_on_target"] = target
            performance_dict["use_ukernel"] = True
            performance_dict["use_chess_for_ukernel"] = chess_for_ukernel
            performance_tests.append(performance_dict)

        performance_tests += [
            {
                "M": 512,
                "N": 512,
                "K": 4096,
                "peano_opt_level": 2,
                "outline": "none",
            },
            {
                "M": 512,
                "N": 512,
                "K": 4096,
                "peano_opt_level": 2,
            },
            {
                "M": 512,
                "N": 512,
                "K": 4096,
                "outline": "none",
            },
            {
                "M": 512,
                "N": 512,
                "K": 4096,
            },
            {
                "M": 512,
                "N": 512,
                "K": 4096,
                "use_ukernel": True,
                "use_chess_for_ukernel": True,
            },
            {
                "M": 512,
                "N": 512,
                "K": 4096,
                "use_packet_flow": True,
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "use_ukernel": True,
                "use_chess_for_ukernel": True,
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "transpose_b": True,
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "use_packet_flow": True,
            },
            {
                "M": 4096,
                "N": 512,
                "K": 512,
            },
            {
                "M": 4096,
                "N": 512,
                "K": 512,
                "use_ukernel": True,
                "use_chess_for_ukernel": True,
            },
            {
                "M": 4096,
                "N": 512,
                "K": 512,
                "transpose_a": True,
            },
            {
                "M": 4096,
                "N": 512,
                "K": 512,
                "use_packet_flow": True,
            },
            {
                "M": 4096,
                "N": 512,
                "K": 512,
                # call_replication = 0 means the compute is omitted, this should
                # help triangulate how much performance gain can be obtained with better
                # matmul on core vs data movement.
                "call_replication": 0,
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "tile_pipeline": "pack-peel-4-level-tiling",
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "use_ukernel": True,
                "tile_pipeline": "pack-peel-4-level-tiling",
                "use_chess_for_ukernel": True,
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "use_ukernel": True,
                "matmul4d": True,
                "tile_pipeline": "pack-peel-4-level-tiling",
                "use_chess_for_ukernel": True,
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "call_replication": 0,
                "tile_pipeline": "pack-peel-4-level-tiling",
            },
            ##############
            # NPU4 Tests #
            ##############
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "in_dtype": "i8",
                "use_ukernel": True,
                "outline": "all",
                "run_on_target": "npu4",
                "stack_size": 3072,
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "in_dtype": "i8",
                "use_ukernel": True,
                "outline": "all",
                "run_on_target": "npu4",
                "use_packet_flow": True,
                "stack_size": 3072,
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "in_dtype": "i8",
                "outline": "all",
                "call_replication": 0,
                "run_on_target": "npu4",
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "in_dtype": "i8",
                "use_ukernel": True,
                "matmul4d": True,
                "scale_trunc": True,
                "tile_pipeline": "pack-peel-4-level-tiling",
                "run_on_target": "npu4",
                "use_chess_for_ukernel": False,
                "stack_size": 3072,
            },
            {
                "M": 512,
                "N": 4096,
                "K": 512,
                "in_dtype": "i8",
                "use_ukernel": True,
                "outline": "all",
                "tile_pipeline": "pack-peel-4-level-tiling",
                "run_on_target": "npu4",
                "stack_size": 3072,
            },
            {
                "M": 1024,
                "N": 4096 * 4,
                "K": 512,
                "in_dtype": "i8",
                "use_ukernel": True,
                "matmul4d": True,
                "scale_trunc": True,
                "tile_pipeline": "pack-peel-4-level-tiling",
                "run_on_target": "npu4",
                "use_chess": True,
                "use_chess_for_ukernel": True,
                "peano_opt_level": 1,
            },
            {
                "M": 1024,
                "N": 4096 * 4,
                "K": 512,
                "in_dtype": "i8",
                "use_ukernel": True,
                "matmul4d": True,
                "scale_trunc": True,
                "tile_pipeline": "pack-peel-4-level-tiling",
                "run_on_target": "npu4",
                "use_chess": False,
                "use_chess_for_ukernel": False,
                "peano_opt_level": 1,
                "stack_size": 3072,
            },
        ]

        # Some bf16 Performance tests:
        for test in performance_tests:
            M = test["M"]
            N = test["N"]
            K = test["K"]

            peano_opt_level = test.get("peano_opt_level", 3)
            outline = test.get("outline", "balanced")
            transpose_a = test.get("transpose_a", False)
            transpose_b = test.get("transpose_b", False)
            use_ukernel = test.get("use_ukernel", False)
            tile_pipeline = test.get("tile_pipeline", "pack-peel")
            matmul4d = test.get("matmul4d", False)
            scale_trunc = test.get("scale_trunc", False)
            use_chess = test.get("use_chess", False)
            use_chess_for_ukernel = test.get("use_chess_for_ukernel", False)
            run_on_target = test.get("run_on_target", "npu1_4col")
            in_dtype = test.get("in_dtype", "bf16")
            out_dtype = test.get("out_dtype", "f32")
            stack_size = test.get("stack_size", "1024")
            use_packet_flow = test.get("use_packet_flow", False)

            # Default of 1 means that outlined functions are called once at each
            # call site (i.e. normal behaviour).
            call_replication = test.get("call_replication", 1)

            if in_dtype == "i8" and out_dtype == "f32":
                out_dtype = "i32"

            n_performance_repeats = test.get("n_performance_repeats", 5)
            n_performance_kernel_runs = test.get("n_performance_kernel_runs", 100)
            additional_labels = test.get("additional_labels", [])

            skip_numerics = call_replication != 1

            outlining_string = "--iree-amdaie-enable-function-outlining=" + outline

            peano_opt_level_string = f'"-O{peano_opt_level}"'
            name_suffix = "O" + str(peano_opt_level)
            name_suffix += "_" + run_on_target

            aie_compilation_flags = test.get("aie_compilation_flags", [])
            aie_compilation_flags += [
                outlining_string,
                f"--iree-amd-aie-additional-peano-opt-flags={peano_opt_level_string}",
            ]

            if call_replication != 1:
                aie_compilation_flags.append(
                    f"--iree-amdaie-call-replication={call_replication}"
                )
                name_suffix += "_callrepl_" + str(call_replication)

            if outline != "none":
                name_suffix += "_outline"

            n_rows = 0
            n_cols = 0
            if run_on_target == "npu1_4col":
                n_rows = 4
                n_cols = 4
            elif run_on_target == "npu4":
                n_rows = 4
                n_cols = 8

            # --iree-amdaie-num-rows=1
            for f in aie_compilation_flags:
                if "--iree-amdaie-num-rows=" in f:
                    n_rows = int(f.split("=")[1])
                if "--iree-amdaie-num-cols=" in f:
                    n_cols = int(f.split("=")[1])

            if n_rows == 0 or n_cols == 0:
                raise ValueError("n_rows or n_cols not set, maybe a new device type?")

            n_matmul_ops = 2 * M * N * K * call_replication

            if matmul4d:
                TestClass = Matmul4d if scale_trunc is False else Matmul4dScaleTrunci
            elif (transpose_a, transpose_b) == (False, False):
                TestClass = Matmul
            elif (transpose_a, transpose_b) == (True, False):
                TestClass = MatmulTransposeA
            elif (transpose_a, transpose_b) == (False, True):
                TestClass = MatmulTransposeB
            else:
                raise ValueError("Transposing both LHS and RHS is not supported.")

            if use_packet_flow:
                # These tests are intended to validate packet flow functionality, so they should make use of as many packet flows as possible.
                # Currently, packet flows are only enabled for kernel inputs to avoid potential deadlocks.
                # TODO(zhewen): Add support for kernel outputs.
                aie_compilation_flags.append(
                    "--iree-amdaie-packet-flow-strategy=inputs"
                )
                name_suffix += "_packet_flow"

            # This should only be the case for benchmark tests which we expect
            # to not pass numerically.
            if skip_numerics:
                pass
            else:
                self.register(
                    TestClass(
                        M,
                        N,
                        K,
                        in_dtype,
                        out_dtype,
                        test_params=TestParams(
                            run_on_target=run_on_target,
                            tile_pipeline=tile_pipeline,
                            use_ukernel=use_ukernel,
                            aie_compilation_flags=aie_compilation_flags,
                            name_suffix=name_suffix,
                            n_repeats=2,
                            use_chess=use_chess,
                            use_chess_for_ukernel=use_chess_for_ukernel,
                            stack_size=stack_size,
                        ),
                        additional_labels=additional_labels,
                    )
                )

            test = TestClass(
                M,
                N,
                K,
                in_dtype,
                out_dtype,
                test_params=TestParams(
                    run_on_target=run_on_target,
                    tile_pipeline=tile_pipeline,
                    use_ukernel=use_ukernel,
                    aie_compilation_flags=aie_compilation_flags,
                    name_suffix=name_suffix,
                    run_benchmark=True,
                    n_repeats=n_performance_repeats,
                    n_kernel_runs=n_performance_kernel_runs,
                    use_chess=use_chess,
                    use_chess_for_ukernel=use_chess_for_ukernel,
                    stack_size=stack_size,
                ),
                additional_labels=additional_labels,
            )
            test.n_matmul_ops = n_matmul_ops
            test.n_rows = n_rows
            test.n_cols = n_cols

            self.register(test)

        # M, K, N (copies from run_matmul_tests.sh)
        bf16_ukernel_shapes_medium = [
            [256, 256, 256],
            [128, 512, 512],
            [512, 4096, 2048],
        ]
        for shape in bf16_ukernel_shapes_medium:
            self.register(
                Matmul(
                    shape[0],
                    shape[2],
                    shape[1],
                    "bf16",
                    "f32",
                    test_params=TestParams(
                        use_ukernel=True,
                        lower_to_aie_pipeline="objectFifo",
                        tile_pipeline="pack-peel",
                        n_repeats=2,
                    ),
                )
            )

        # Control packet single dispatch tests:
        for target, in_type, out_type in [
            ["npu1_4col", "i8", "i32"],
            ["npu4", "i32", "i32"],
        ]:
            # Numeric test for reconfiguration on the whole AIE array.
            self.register(
                Matmul(
                    1024,
                    1024,
                    1024,
                    in_type,
                    out_type,
                    test_params=TestParams(
                        run_on_target=target,
                        enable_ctrlpkt=True,
                        name_suffix=target,
                        n_repeats=10,
                    ),
                )
            )
            # Benchmark reconfiguration time only, do not run the kernel.
            self.register(
                Matmul(
                    1024,
                    1024,
                    1024,
                    in_type,
                    out_type,
                    test_params=TestParams(
                        run_benchmark=True,
                        n_repeats=2,
                        n_kernel_runs=0,
                        n_reconfigure_runs=100,
                        n_pdi_loads=1,
                        run_on_target=target,
                        enable_ctrlpkt=True,
                        name_suffix=target + "_reconfigure_only",
                    ),
                )
            )
            # Benchmark PDI load time only (control packet disabled), do not run the kernel.
            self.register(
                Matmul(
                    1024,
                    1024,
                    1024,
                    in_type,
                    out_type,
                    test_params=TestParams(
                        run_benchmark=True,
                        n_repeats=2,
                        n_kernel_runs=0,
                        n_reconfigure_runs=0,
                        n_pdi_loads=100,
                        run_on_target=target,
                        enable_ctrlpkt=False,
                        name_suffix=target + "_pdi_load_only",
                    ),
                )
            )

        # MultipleDispatches tests:
        for target, enable_ctrlpkt in product(["npu1_4col", "npu4"], [False, True]):
            for file_base_name, func_name in [
                ["two_matmul_switching", "matmul_small"],
                ["matmul_f32_8_8_4", "matmul_8_8_4"],
                ["matmul_f32_8_4_8", "matmul_8_4_8"],
                ["three_matmuls", "three_$mm$"],
            ]:
                self.register(
                    MultipleDispatches(
                        file_base_name,
                        func_name,
                        test_params=TestParams(
                            aie_compilation_flags=[
                                "--iree-amdaie-num-rows=1",
                                "--iree-amdaie-num-cols=1",
                            ],
                            run_on_target=target,
                            name_suffix="OneCore_" + target,
                            enable_ctrlpkt=enable_ctrlpkt,
                            n_repeats=50 if enable_ctrlpkt else 2,
                        ),
                    )
                )

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
            generator = ConvolutionMlirGenerator(**conv_2d_map)
            self.register(
                ConvolutionFromTemplate(
                    generator, TestParams(tile_pipeline="conv-decompose", n_repeats=2)
                )
            )

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
        generator = ConvolutionMlirGenerator(**depthwise_map)
        self.register(
            ConvolutionFromTemplate(
                generator, TestParams(tile_pipeline="conv-decompose", n_repeats=2)
            )
        )

        # Softmax tests:
        # Note: The error tolerance for npu4 is higher than that for npu1_4col.
        # npu1_4col uses a lookup table to compute exponentials,
        # whereas npu4 uses a native exp2 instruction, which is less accurate.
        for target, rtol in [["npu1_4col", 4e-2], ["npu4", 8e-2]]:
            for run_benchmark in [False, True]:
                self.register(
                    Softmax(
                        8192,
                        1024,
                        "bf16",
                        test_params=TestParams(
                            run_on_target=target,
                            name_suffix=target,
                            use_chess=True,
                            use_chess_for_ukernel=True,
                            use_ukernel=True,
                            tile_pipeline="general-copy",
                            run_benchmark=run_benchmark,
                            n_repeats=2,
                            n_kernel_runs=100,
                            rtol=rtol,
                        ),
                    )
                )

        # Reduction op tests:
        self.register(
            Reduction(
                file_base_name="reduction_sum",
                function_name="reduction_sum",
                test_params=TestParams(
                    name_suffix="sum",
                    tile_pipeline="general-copy",
                ),
            )
        )

        # Soak testing.
        # See https://github.com/nod-ai/iree-amd-aie/issues/1264
        seed = 42
        rng = np.random.default_rng(seed=seed)

        # The number of randomly sampled types to test.
        n_runs = 2

        MN_pool = [128, 256, 512]
        K_pool = [128, 256]
        input_type_pool = ["bf16"]

        # The number of possible combinations of the type (M, N, K, input_type)
        grid_elements = len(MN_pool) * len(MN_pool) * len(K_pool) * len(input_type_pool)
        if n_runs >= 0.5 * grid_elements:
            raise RuntimeError(
                "Sampling without replacement nearly exhausted, might as well test full grid"
            )

        # We'll do sampling without replacement (i.e. we won't test the same type more than once).
        sampled = set()

        for run_number in range(n_runs):
            # Generate a new random type.
            while True:
                M = rng.choice(MN_pool)
                N = rng.choice(MN_pool)
                K = rng.choice(K_pool)
                input_type = rng.choice(input_type_pool)
                sample = (M, N, K, input_type)
                if sample not in sampled:
                    sampled.add(sample)
                    break

            # Test the random type.
            output_type = "i32" if input_type == "i8" else "f32"
            self.register(
                Matmul(
                    M,
                    N,
                    K,
                    input_type,
                    output_type,
                    test_params=TestParams(
                        name_suffix="soak_" + str(run_number),
                    ),
                    additional_labels=["Soak"],
                )
            )


def all_tests(
    tests,
    output_dir,
    iree_dir,
    peano_dir,
    xrt_dir,
    vitis_dir,
    verbose,
    reset_npu_between_runs,
    do_not_run_aie,
    test_set,
    skip_test_set,
    device_hal,
    xrt_lite_n_core_rows,
    xrt_lite_n_core_cols,
    target_device,
):
    """
    There are a few ways to add tests to this script:

    1) add a single test file in `./test_files` which should follow the same
       format as the example `./test_files/matmul_int32.mlir`.

    2) use an existing template in `./matmul_template` to generate a test file
       with a fixed structure. Currently a handful of matmul templates exist in
       that directory.

    3) create a new matmul template in `./matmul_template`, for example if you
       want to add a new variant with transposed operands or unary elementwise
       operations.

    4) create a new template generator, duplicating the directory structure of
       ./matmul_template. For example you might want to create ./conv_template
    """

    if not output_dir.exists():
        output_dir.mkdir()
    if not iree_dir.exists():
        raise RuntimeError(f"'{iree_dir}' is not a directory.")
    iree_benchmark_exe = find_executable(iree_dir, "iree-benchmark-module")
    iree_compile_exe = find_executable(iree_dir, "iree-compile")
    iree_run_exe = find_executable(iree_dir, "iree-run-module")
    file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    config = TestConfig(
        output_dir,
        iree_dir,
        peano_dir,
        xrt_dir,
        vitis_dir,
        file_dir,
        iree_benchmark_exe,
        iree_compile_exe,
        iree_run_exe,
        verbose,
        reset_npu_between_runs,
        do_not_run_aie,
        device_hal,
        xrt_lite_n_core_rows,
        xrt_lite_n_core_cols,
        target_device,
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
        skip = test.name in skip_test_set or any(
            (label in skip_test_set for label in test.labels)
        )
        if skip:
            not_match.append(test.name)
            continue

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

    parser.add_argument(
        "output_dir",
        type=abs_path,
        help="The directory where artifacts are saved. Each test will have a subdirectory in this directory based on its (unique) name, containing all test specific files.",
    )

    parser.add_argument(
        "iree_dir",
        type=abs_path,
        help="Either the build or install directory of IREE. iree-compile will be looked for in a `bin` or `tools` subdirectory.",
    )

    parser.add_argument(
        "--peano_dir",
        type=abs_path,
        help="The directory where peano is installed. Typically a directory called `llvm-aie`, and obtained as a wheel.",
    )

    parser.add_argument(
        "--vitis_dir",
        type=abs_path,
        help="The directory where the vitis project is installed. This path must be provided if the chess compiler is to be used.",
    )

    parser.add_argument("--xrt_dir", type=abs_path)
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
        "--reset_npu_between_runs",
        action="store_true",
        help=(
            "If passed then the NPU is not reset between runs, otherwise it is reset. "
            "Resetting between runs can in theory help avoid certain types of "
            "errors in parts of the stack which these tests are not designed to catch."
        ),
    )

    parser.add_argument(
        "--do_not_run_aie",
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
        "--aie_compilation_flags",
        type=str,
        help=dedent(
            """
            Additional flags to pass to the AIE compiler, for all tests.
            Example, to print the IR between passes during compilation you might have:
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

    tests_help_string = (
        "A comma-separated list of test names or sets to run. Available test sets: "
        + f"{label_string}"
        + f". Available individual tests: {name_string}. "
    )
    skip_tests_help_string = (
        "A comma-separated list of test names or sets to skip. Available test sets: "
        + f"{label_string}"
        + f". Available individual tests: {name_string}. "
    )

    parser.add_argument(
        "--tests",
        type=str,
        help=tests_help_string,
        default="All",
    )

    parser.add_argument(
        "--skip_tests",
        type=str,
        help=skip_tests_help_string,
        default="",
    )

    parser.add_argument(
        "--device_hal",
        default="xrt-lite",
        const="xrt-lite",
        nargs="?",
        choices=["xrt", "xrt-lite"],
        help="device HAL to use (default: %(default)s)",
    )

    parser.epilog = "Example call: ./run.py --verbose  output_dir ${IREE_INSTALL}  --peano_dir=${LLVM_AIE}  --target_device=npu1_4col --xrt_lite_n_core_rows=4 --xrt_lite_n_core_cols=4 --tests=Peano"

    args = parser.parse_args()

    test_set_list = args.tests.split(",")
    skip_test_list = args.skip_tests.split(",")

    if args.target_device not in current_devices:
        raise ValueError(
            f"Invalid target device '{args.target_device}'. Available options: {current_devices}"
        )
    tests.add_aie_compilation_flags(args.aie_compilation_flags)

    # At least one of peano_dir and vitis_dir must be provided:
    if not args.peano_dir and not args.vitis_dir:
        raise ValueError(
            "At least one of --peano_dir and --vitis_dir must be provided to run the tests."
        )

    all_tests(
        tests,
        args.output_dir,
        args.iree_dir,
        args.peano_dir,
        args.xrt_dir,
        args.vitis_dir,
        args.verbose,
        args.reset_npu_between_runs,
        args.do_not_run_aie,
        test_set_list,
        skip_test_list,
        args.device_hal,
        args.xrt_lite_n_core_rows,
        args.xrt_lite_n_core_cols,
        args.target_device,
    )
