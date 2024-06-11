import os
from pathlib import Path

import numpy as np
import basic_matrix_multiplication_matrix_vector
import basic_matrix_multiplication_single_core

os.environ["VITIS"] = "/opt/tools/Xilinx/Vitis/2023.2"

from iree.compiler import compile_file

# don't forget LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/usr/lib/x86_64-linux-gnu
RUN = True
if RUN:
    from filelock import FileLock
    from xaiepy.xrt import XCLBin


TEMPLATE = """
module attributes {hal.device.targets = [#hal.device.target<"amd-aie-direct", [#hal.executable.target<"amd-aie-direct", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>]>]} {
  hal.executable private @dummy1 {
    hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie-direct", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
      hal.executable.export public @dummy2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.MODULE
    }
  }
  util.func public @dummy3(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = ""}} {
    // this is all gibberish just to hit serializeExecutable
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %element_type_i8 = hal.element_type<i8> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c1, %c1]) type(%element_type_i8) encoding(%dense_row_major)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<1024x512xi8> in !stream.resource<external>{%c1}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<external>{%c1} => !stream.timepoint

    %2 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c1}) {
      stream.cmd.dispatch @dummy1::@amdaie_xclbin_fb::@dummy2 {
        ro %arg2[%c0 for %c1] : !stream.resource<external>{%c1}
      }
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %result : !stream.resource<external>{%c1}
    %4 = stream.tensor.export %3 : tensor<1024x1024xi32> in !stream.resource<external>{%c1} -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
}
"""


def compile(workdir, test):
    compile_file(
        str(workdir / (test + ".mlir")),
        target_backends=["amd-aie-direct"],
        extra_args=[
            "--compile-mode=hal-executable",
            f"--iree-hal-dump-executable-intermediates-to={workdir}",
        ],
    )


def test_matrix_vector_32_1_core():
    M = K = 32
    TEST = basic_matrix_multiplication_matrix_vector.__name__ + "_32_1_core"
    WORKDIR = Path(__file__).parent.absolute() / TEST
    if not WORKDIR.exists():
        WORKDIR.mkdir(parents=True)
    with open(WORKDIR / f"{TEST}.mlir", "w") as f:
        f.write(
            TEMPLATE.replace(
                "MODULE",
                basic_matrix_multiplication_matrix_vector.emit_module(M, K),
            )
        )

    NPU_INSTS_FP = f"{WORKDIR}/module_dummy1_amdaie_xclbin_fb/module_dummy1_amdaie_xclbin_fb.npu.txt"
    XCLBIN_PATH = f"{WORKDIR}/module_dummy1_amdaie_xclbin_fb/module_dummy1_amdaie_xclbin_fb.xclbin"
    KERNEL_NAME = "dummy2"

    compile(WORKDIR, TEST)

    with open(NPU_INSTS_FP, "r") as f:
        npu_insts = list(map(lambda n: int(n, 16), f.readlines()))

    if RUN:
        with FileLock("/tmp/npu.lock"):
            xclbin = XCLBin(XCLBIN_PATH, KERNEL_NAME)
            views = xclbin.mmap_buffers([(M, K), (K,), (M,)], np.float32)

            xclbin.load_npu_instructions(npu_insts)

            A = np.random.randint(0, 10, (M, K)).astype(np.float32)
            B = np.random.randint(0, 10, (K,)).astype(np.float32)
            C = np.zeros((M,)).astype(np.float32)

            wraps = list(map(np.asarray, views))
            np.copyto(wraps[0], A, casting="no")
            np.copyto(wraps[1], B, casting="no")
            np.copyto(wraps[2], C, casting="no")

            xclbin.sync_buffers_to_device()
            xclbin.run()
            print("Running kernel")
            xclbin.wait(30)
            xclbin.sync_buffers_from_device()

            assert np.allclose(A @ B, wraps[2])
            print(wraps[2])


def test_matrix_vector_64_1_core():
    M = K = 64
    TEST = basic_matrix_multiplication_matrix_vector.__name__ + "_64_1_core"
    WORKDIR = Path(__file__).parent.absolute() / TEST
    if not WORKDIR.exists():
        WORKDIR.mkdir(parents=True)
    with open(WORKDIR / f"{TEST}.mlir", "w") as f:
        f.write(
            TEMPLATE.replace(
                "MODULE", basic_matrix_multiplication_matrix_vector.emit_module(M, K)
            )
        )

    NPU_INSTS_FP = f"{WORKDIR}/module_dummy1_amdaie_xclbin_fb/module_dummy1_amdaie_xclbin_fb.npu.txt"
    XCLBIN_PATH = f"{WORKDIR}/module_dummy1_amdaie_xclbin_fb/module_dummy1_amdaie_xclbin_fb.xclbin"
    KERNEL_NAME = "dummy2"

    compile(WORKDIR, TEST)

    with open(NPU_INSTS_FP, "r") as f:
        npu_insts = list(map(lambda n: int(n, 16), f.readlines()))

    if RUN:
        with FileLock("/tmp/npu.lock"):
            xclbin = XCLBin(XCLBIN_PATH, KERNEL_NAME)
            views = xclbin.mmap_buffers([(M, K), (K,), (M,)], np.float32)

            xclbin.load_npu_instructions(npu_insts)

            A = np.random.randint(0, 10, (M, K)).astype(np.float32)
            B = np.random.randint(0, 10, (K,)).astype(np.float32)
            C = np.zeros((M,)).astype(np.float32)

            wraps = list(map(np.asarray, views))
            np.copyto(wraps[0], A, casting="no")
            np.copyto(wraps[1], B, casting="no")
            np.copyto(wraps[2], C, casting="no")

            xclbin.sync_buffers_to_device()
            xclbin.run()
            print("Running kernel")
            xclbin.wait(30)
            xclbin.sync_buffers_from_device()

            assert np.allclose(A @ B, wraps[2])
            print(wraps[2])


def test_matrix_vector_2_cores():
    M = K = 64
    TEST = basic_matrix_multiplication_matrix_vector.__name__ + "_64_2_cores"
    WORKDIR = Path(__file__).parent.absolute() / TEST
    if not WORKDIR.exists():
        WORKDIR.mkdir(parents=True)
    with open(WORKDIR / f"{TEST}.mlir", "w") as f:
        f.write(
            TEMPLATE.replace(
                "MODULE",
                basic_matrix_multiplication_matrix_vector.emit_module(M, K, n_cores=2),
            )
        )

    NPU_INSTS_FP = f"{WORKDIR}/module_dummy1_amdaie_xclbin_fb/module_dummy1_amdaie_xclbin_fb.npu.txt"
    XCLBIN_PATH = f"{WORKDIR}/module_dummy1_amdaie_xclbin_fb/module_dummy1_amdaie_xclbin_fb.xclbin"
    KERNEL_NAME = "dummy2"

    compile(WORKDIR, TEST)

    with open(NPU_INSTS_FP, "r") as f:
        npu_insts = list(map(lambda n: int(n, 16), f.readlines()))

    if RUN:
        with FileLock("/tmp/npu.lock"):
            xclbin = XCLBin(XCLBIN_PATH, KERNEL_NAME)
            views = xclbin.mmap_buffers([(M, K), (K,), (M,)], np.float32)

            xclbin.load_npu_instructions(npu_insts)

            A = np.random.randint(0, 10, (M, K)).astype(np.float32)
            B = np.random.randint(0, 10, (K,)).astype(np.float32)
            C = np.zeros((M,)).astype(np.float32)

            wraps = list(map(np.asarray, views))
            np.copyto(wraps[0], A, casting="no")
            np.copyto(wraps[1], B, casting="no")
            np.copyto(wraps[2], C, casting="no")

            xclbin.sync_buffers_to_device()
            xclbin.run()
            print("Running kernel")
            xclbin.wait(30)
            xclbin.sync_buffers_from_device()

            assert np.allclose(A @ B, wraps[2])
            print(wraps[2])


def test_matmul_32():
    M = K = N = 32
    TEST = basic_matrix_multiplication_single_core.__name__ + "_32"
    WORKDIR = Path(__file__).parent.absolute() / TEST
    if not WORKDIR.exists():
        WORKDIR.mkdir(parents=True)
    with open(WORKDIR / f"{TEST}.mlir", "w") as f:
        f.write(
            TEMPLATE.replace(
                "MODULE", basic_matrix_multiplication_single_core.emit_module(M, K, N)
            )
        )
    NPU_INSTS_FP = f"{WORKDIR}/module_dummy1_amdaie_xclbin_fb/module_dummy1_amdaie_xclbin_fb.npu.txt"
    XCLBIN_PATH = f"{WORKDIR}/module_dummy1_amdaie_xclbin_fb/module_dummy1_amdaie_xclbin_fb.xclbin"
    KERNEL_NAME = "dummy2"

    compile(WORKDIR, TEST)

    with open(NPU_INSTS_FP, "r") as f:
        npu_insts = list(map(lambda n: int(n, 16), f.readlines()))

    if RUN:
        with FileLock("/tmp/npu.lock"):
            xclbin = XCLBin(XCLBIN_PATH, KERNEL_NAME)
            views = xclbin.mmap_buffers([(M, K), (K, N), (M, N)], np.float32)

            xclbin.load_npu_instructions(npu_insts)

            # the stupid upstream example isn't correct for real numbers
            A = np.ones((M, K)).astype(np.float32)
            B = 2 * np.ones((K, N)).astype(np.float32)
            C = np.zeros((M, N)).astype(np.float32)

            wraps = list(map(np.asarray, views))
            np.copyto(wraps[0], A, casting="no")
            np.copyto(wraps[1], B, casting="no")
            np.copyto(wraps[2], C, casting="no")

            xclbin.sync_buffers_to_device()
            xclbin.run()
            print("Running kernel")
            xclbin.wait(30)
            xclbin.sync_buffers_from_device()

            assert np.allclose(A @ B, wraps[2])
            print(wraps[2])


def test_matmul_64():
    M = K = N = 64
    TEST = basic_matrix_multiplication_single_core.__name__ + "_64"
    WORKDIR = Path(__file__).parent.absolute() / TEST
    if not WORKDIR.exists():
        WORKDIR.mkdir(parents=True)
    with open(WORKDIR / f"{TEST}.mlir", "w") as f:
        f.write(
            TEMPLATE.replace(
                "MODULE", basic_matrix_multiplication_single_core.emit_module(M, K, N)
            )
        )
    NPU_INSTS_FP = f"{WORKDIR}/module_dummy1_amdaie_xclbin_fb/module_dummy1_amdaie_xclbin_fb.npu.txt"
    XCLBIN_PATH = f"{WORKDIR}/module_dummy1_amdaie_xclbin_fb/module_dummy1_amdaie_xclbin_fb.xclbin"
    KERNEL_NAME = "dummy2"

    compile(WORKDIR, TEST)

    with open(NPU_INSTS_FP, "r") as f:
        npu_insts = list(map(lambda n: int(n, 16), f.readlines()))

    if RUN:
        with FileLock("/tmp/npu.lock"):
            xclbin = XCLBin(XCLBIN_PATH, KERNEL_NAME)
            views = xclbin.mmap_buffers([(M, K), (K, N), (M, N)], np.float32)

            xclbin.load_npu_instructions(npu_insts)

            # the stupid upstream example isn't correct for real numbers
            A = np.ones((M, K)).astype(np.float32)
            B = 2 * np.ones((K, N)).astype(np.float32)
            C = np.zeros((M, N)).astype(np.float32)

            wraps = list(map(np.asarray, views))
            np.copyto(wraps[0], A, casting="no")
            np.copyto(wraps[1], B, casting="no")
            np.copyto(wraps[2], C, casting="no")

            xclbin.sync_buffers_to_device()
            xclbin.run()
            print("Running kernel")
            xclbin.wait(30)
            xclbin.sync_buffers_from_device()

            assert np.allclose(A @ B, wraps[2])
            print(wraps[2])
