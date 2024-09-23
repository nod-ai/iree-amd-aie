import os
import pathlib

import numpy as np

from iree import compiler as ireec
from iree import runtime as ireert
from iree.compiler import ir
from iree.compiler.dialects import arith, tensor, linalg
from iree.compiler.dialects.builtin import module
from iree.compiler.dialects.func import func
from iree.compiler.extras import types as T
from iree.runtime import get_driver

with ir.Context(), ir.Location.unknown():

    @module(sym_name="arithmetic")
    def arithmetic():
        @func(T.tensor(32, 16, T.i8()), T.tensor(16, 32, T.i8()))
        def matmul_i8_i32(lhs, rhs):
            cst = arith.constant(T.i32(), 0)
            v0 = tensor.empty([32, 32], T.i32())
            v1 = linalg.fill(cst, outs=[v0])
            return linalg.matmul(lhs, rhs, outs=[v1])

    print(arithmetic)

TARGET_BACKEND = "amd-aie"
WORK_DIR = pathlib.Path(__file__).cwd() / "executable_cache_test"
WORK_DIR = WORK_DIR.absolute()
with ireec.tools.TempFileSaver(str(WORK_DIR)):
    compiled_flatbuffer = ireec.tools.compile_str(
        str(arithmetic),
        target_backends=[TARGET_BACKEND],
        extra_args=[
            f"--iree-hal-dump-executable-files-to={WORK_DIR}",
            f"--iree-hal-target-backends={TARGET_BACKEND}",
            "--iree-amdaie-lower-to-aie-pipeline=air",
            f"--iree-amd-aie-peano-install-dir={os.getenv('PEANO_INSTALL_DIR')}",
            f"--iree-amd-aie-install-dir={os.getenv('IREE_INSTALL_DIR')}",
        ],
    )

driver = get_driver("xrt")

config = ireert.Config(device=driver.create_default_device())
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
ctx.add_vm_module(vm_module)

# Invoke the function and print the result.
arg0 = np.ones((32, 16), dtype=np.int8)
arg1 = np.ones((16, 32), dtype=np.int8)
f = ctx.modules.arithmetic["matmul_i8_i32"]
results = f(arg0, arg1).to_host()
print("Results:", results)
