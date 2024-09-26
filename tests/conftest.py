import os
from contextlib import contextmanager

import numpy as np
import pytest
from iree.runtime import VmModule

from iree.compiler import ir
from iree.compiler._mlir_libs import get_dialect_registry
from iree.compiler.api import Session, Output, Source
from iree.compiler.extras import types as T
from iree.runtime import get_driver, Config, SystemContext


for t in [
    "i8",
    "i16",
    "i32",
    "i64",
    "si8",
    "si16",
    "si32",
    "si64",
    "ui8",
    "ui16",
    "ui32",
    "ui64",
    "f16",
    "f32",
    "f64",
    "bf16",
]:
    tf = getattr(T, t)
    tf.__name__ = t


@pytest.fixture
def iree_session(request) -> Session:
    s = Session()
    s.context.append_dialect_registry(get_dialect_registry())
    s.context.load_all_available_dialects()
    target_backend = getattr(request, "target_backend", "amd-aie")
    pipeline = getattr(request, "pipeline", "air")
    s.set_flags(
        f"--iree-hal-target-backends={target_backend}",
        # TODO(max): normalize iree-amdaie/iree-amd-aie in pass strings
        f"--iree-amdaie-lower-to-aie-pipeline={pipeline}",
        f"--iree-amd-aie-peano-install-dir={os.getenv('PEANO_INSTALL_DIR')}",
        f"--iree-amd-aie-install-dir={os.getenv('IREE_INSTALL_DIR')}",
    )
    yield s


@pytest.fixture
def session_module(iree_session, tmp_path) -> ir.Module:
    iree_session.set_flags(
        f"--iree-hal-dump-executable-files-to={tmp_path}",
    )
    with ir.Location.unknown(iree_session.context):
        module_op = ir.Module.create()
        with ir.InsertionPoint(module_op.body):
            yield iree_session, module_op


@contextmanager
def invokable_module(session, module, device="xrt") -> VmModule:
    source = Source.wrap_buffer(session, str(module).encode())
    inv = session.invocation()
    inv.parse_source(source)
    inv.execute()
    compiled_flatbuffer = Output.open_membuffer()
    inv.output_vm_bytecode(compiled_flatbuffer)

    driver = get_driver(device)
    config = Config(device=driver.create_default_device())
    ctx = SystemContext(config=config)
    vm_module = VmModule.copy_buffer(ctx.instance, compiled_flatbuffer.map_memory())
    ctx.add_vm_module(vm_module)

    try:
        yield ctx.modules.module
    finally:
        inv.close()


_np_dtype_to_mlir_type_ctor = {
    np.int8: T.i8,
    np.int16: T.i16,
    np.int32: T.i32,
    # windows
    np.intc: T.i32,
    np.int64: T.i64,
    # is technically wrong i guess but numpy by default casts python scalars to this
    # so to support passing lists of ints we map to index type
    np.longlong: T.index,
    np.uintp: T.index,
    np.float16: T.f16,
    np.float32: T.f32,
    np.float64: T.f64,
}

_mlir_type_ctor_to_np_dtype = lambda: {
    v: k for k, v in _np_dtype_to_mlir_type_ctor.items()
}


def np_dtype_to_mlir_type(np_dtype):
    if typ := _np_dtype_to_mlir_type_ctor.get(np_dtype):
        return typ()


def mlir_type_to_np_dtype(mlir_type):
    _mlir_type_to_np_dtype = {v(): k for k, v in _np_dtype_to_mlir_type_ctor.items()}
    return _mlir_type_to_np_dtype.get(mlir_type)
