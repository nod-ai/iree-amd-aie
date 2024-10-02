from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from ml_dtypes import bfloat16

from iree.compiler import ir
from iree.compiler._mlir_libs import get_dialect_registry
from iree.compiler.api import Session, Output, Source, _initializeGlobalCL
from iree.compiler.extras import types as T
from iree.runtime import VmModule
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


def ids(datum):
    if callable(datum):
        return datum.__name__
    return datum


def pytest_addoption(parser):
    abs_path = lambda x: Path(x).absolute()
    parser.addoption("--iree-install-dir", type=abs_path, required=True)
    parser.addoption("--peano-install-dir", type=abs_path)
    parser.addoption("--output-dir", type=abs_path)
    parser.addoption("--vitis-dir", type=abs_path)
    parser.addoption("--iree-aie-debug", action="store_true")


@pytest.fixture(scope="session")
def global_cl_args(request):
    _initializeGlobalCL(
        "--iree-hal-memoization=false",
        "--iree-hal-indirect-command-buffers=false",
    )


@pytest.fixture
def iree_session(request, pytestconfig, global_cl_args) -> Session:
    s = Session()
    s.context.append_dialect_registry(get_dialect_registry())
    s.context.load_all_available_dialects()
    target_backend = request.node.callspec.params.get("target_backend", "amd-aie")
    target_device = request.node.callspec.params.get("target_device", "npu1_4col")
    lower_to_aie_pipeline = request.node.callspec.params.get(
        "lower_to_aie_pipeline", "air"
    )
    tile_pipeline = request.node.callspec.params.get("tile_pipeline", "pad-pack")
    use_chess = request.node.callspec.params.get("use_chess", False)
    enable_packet_flow = request.node.callspec.params.get("enable_packet_flow", False)
    # TODO(max): normalize iree-amdaie/iree-amd-aie in pass strings
    flags = [
        f"--iree-hal-target-backends={target_backend}",
        f"--iree-amdaie-target-device={target_device}",
        f"--iree-amdaie-lower-to-aie-pipeline={lower_to_aie_pipeline}",
        f"--iree-amdaie-tile-pipeline={tile_pipeline}",
        f"--iree-amd-aie-peano-install-dir={pytestconfig.option.peano_install_dir}",
        f"--iree-amd-aie-install-dir={pytestconfig.option.iree_install_dir}",
        f"--iree-amd-aie-enable-chess={use_chess}",
        f"--iree-amdaie-enable-packet-flow={enable_packet_flow}",
    ]
    if pytestconfig.option.vitis_dir:
        flags += [f"--iree-amd-aie-vitis-install-dir={pytestconfig.option.vitis_dir}"]
    if pytestconfig.option.iree_aie_debug:
        flags += [
            "--iree-amd-aie-show-invoked-commands",
            "--aie2xclbin-print-ir-after-all",
        ]
    if pytestconfig.option.output_dir:
        flags += [
            f"--iree-hal-dump-executable-files-to={pytestconfig.option.output_dir}"
        ]

    s.set_flags(*flags)
    yield s


@pytest.fixture
def session_module(iree_session, tmp_path) -> ir.Module:
    with ir.Location.unknown(iree_session.context):
        module_op = ir.Module.create()
        with ir.InsertionPoint(module_op.body):
            yield iree_session, module_op


@pytest.fixture(scope="session")
def device(device="xrt") -> ir.Module:
    yield get_driver(device).create_default_device()


@contextmanager
def invokable_module(session, module, device) -> VmModule:
    source = Source.wrap_buffer(session, str(module).encode())
    inv = session.invocation()
    inv.parse_source(source)
    inv.execute()
    compiled_flatbuffer = Output.open_membuffer()
    inv.output_vm_bytecode(compiled_flatbuffer)

    config = Config(device=device)
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
    bfloat16: T.bf16,
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
