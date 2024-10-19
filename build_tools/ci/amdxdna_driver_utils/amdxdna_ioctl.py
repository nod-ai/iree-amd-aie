import argparse
import array
import ctypes
import ctypes.util
import fcntl
import pathlib
import re
import struct
from argparse import Namespace
from pprint import pformat

import amdxdna_accel
from amdxdna_accel import (
    struct_amdxdna_drm_query_aie_version,
    struct_amdxdna_drm_get_info,
    struct_amdxdna_drm_query_aie_metadata,
    DRM_AMDXDNA_QUERY_AIE_VERSION,
    DRM_AMDXDNA_QUERY_AIE_METADATA,
)

_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2

_IOC_NRMASK = (1 << _IOC_NRBITS) - 1
_IOC_TYPEMASK = (1 << _IOC_TYPEBITS) - 1
_IOC_SIZEMASK = (1 << _IOC_SIZEBITS) - 1
_IOC_DIRMASK = (1 << _IOC_DIRBITS) - 1

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

IOC_NONE = 0
IOC_WRITE = 1
IOC_READ = 2


def _IOC(dir, type, nr, size):
    assert dir <= _IOC_DIRMASK, dir
    assert type <= _IOC_TYPEMASK, type
    assert nr <= _IOC_NRMASK, nr
    assert size <= _IOC_SIZEMASK, size
    return (
        (dir << _IOC_DIRSHIFT)
        | (type << _IOC_TYPESHIFT)
        | (nr << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


def _IOC_TYPECHECK(t):
    if isinstance(t, (memoryview, bytearray)):
        size = len(t)
    elif isinstance(t, struct.Struct):
        size = t.size
    elif isinstance(t, array.array):
        size = t.itemsize * len(t)
    else:
        size = ctypes.sizeof(t)
    assert size <= _IOC_SIZEMASK, size
    return size


def _IOWR(type, nr, size):
    return _IOC(IOC_READ | IOC_WRITE, type, nr, _IOC_TYPECHECK(size))


def get_struct(argp, stype):
    return ctypes.cast(ctypes.c_void_p(argp), ctypes.POINTER(stype)).contents


def get_void_ptr_to_struct(s):
    ptr = ctypes.pointer(s)
    return ctypes.cast(ptr, ctypes.c_void_p)


def format_struct(s):
    return pformat(s.as_dict(s))


# <drm.h>
DRM_IOCTL_BASE = ord("d")
DRM_COMMAND_BASE = 0x40


def DRM_IOWR(nr, type):
    return _IOWR(DRM_IOCTL_BASE, nr, type)


def ioctls_from_header():
    hdr = (
        (pathlib.Path(__file__).parent / "amdxdna_accel.py")
        .read_text()
        .replace("\\\n", "")
    )
    pattern = "DRM_IOCTL_AMDXDNA_([A-Z0-9_]+) = DRM_IOWR \( DRM_COMMAND_BASE \+ DRM_AMDXDNA_([A-Z0-9_]+) , struct_amdxdna_drm_([a-z0-9_]+) \)"
    matches = re.findall(pattern, hdr, re.MULTILINE)
    ioctls = Namespace()
    for name, offset, sname in matches:
        assert name == offset
        offset = f"DRM_AMDXDNA_{name}"
        assert hasattr(amdxdna_accel, offset)
        offset = getattr(amdxdna_accel, offset)
        struc = getattr(amdxdna_accel, "struct_amdxdna_drm_" + sname)
        setattr(
            ioctls,
            f"DRM_IOCTL_AMDXDNA_{name}",
            DRM_IOWR(DRM_COMMAND_BASE + offset, struc),
        )

    return ioctls


ioctls = ioctls_from_header()


def get_aie_version(drv_fd):
    version = struct_amdxdna_drm_query_aie_version()
    info_params = struct_amdxdna_drm_get_info(
        DRM_AMDXDNA_QUERY_AIE_VERSION,
        ctypes.sizeof(struct_amdxdna_drm_query_aie_version),
        get_void_ptr_to_struct(version).value,
    )

    fcntl.ioctl(drv_fd, ioctls.DRM_IOCTL_AMDXDNA_GET_INFO, info_params)

    return version.major, version.minor


def get_aie_metadata(drv_fd):
    metadata = struct_amdxdna_drm_query_aie_metadata()
    info_params = struct_amdxdna_drm_get_info(
        DRM_AMDXDNA_QUERY_AIE_METADATA,
        ctypes.sizeof(struct_amdxdna_drm_query_aie_metadata),
        get_void_ptr_to_struct(metadata).value,
    )

    fcntl.ioctl(drv_fd, ioctls.DRM_IOCTL_AMDXDNA_GET_INFO, info_params)

    return format_struct(metadata)


def get_core_n_rows(drv_fd):
    metadata = struct_amdxdna_drm_query_aie_metadata()
    info_params = struct_amdxdna_drm_get_info(
        DRM_AMDXDNA_QUERY_AIE_METADATA,
        ctypes.sizeof(struct_amdxdna_drm_query_aie_metadata),
        get_void_ptr_to_struct(metadata).value,
    )

    fcntl.ioctl(drv_fd, ioctls.DRM_IOCTL_AMDXDNA_GET_INFO, info_params)
    return metadata.core.row_count


def find_npu_device():
    drvpath = pathlib.Path("/sys/bus/pci/drivers/amdxdna")
    for file in drvpath.iterdir():
        if file.is_symlink():
            actual_path = (drvpath / file.readlink()).resolve()
            if str(actual_path).startswith("/sys/devices/pci"):
                return actual_path
    raise RuntimeError("npu device not found")


def read_vbnv(npu_device_path):
    f = open(npu_device_path / "vbnv")
    vbnv = f.read()
    assert vbnv.startswith("RyzenAI-")
    return vbnv.split("-")[-1].strip()


def get_core_n_cols(drv_fd, npu_device):
    metadata = struct_amdxdna_drm_query_aie_metadata()
    info_params = struct_amdxdna_drm_get_info(
        DRM_AMDXDNA_QUERY_AIE_METADATA,
        ctypes.sizeof(struct_amdxdna_drm_query_aie_metadata),
        get_void_ptr_to_struct(metadata).value,
    )

    fcntl.ioctl(drv_fd, ioctls.DRM_IOCTL_AMDXDNA_GET_INFO, info_params)
    if npu_device == "npu1":
        # phoenix
        return metadata.cols - 1
    elif npu_device == "npu4":
        # strix
        return metadata.cols

    return NotImplementedError(f"unrecognized {npu_device=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu-device", action="store_true")
    parser.add_argument("--num-rows", action="store_true")
    parser.add_argument("--num-cols", action="store_true")
    parser.add_argument("--aie-metadata", action="store_true")
    parser.add_argument("--aie-version", action="store_true")
    args = parser.parse_args()

    drv_path = "/dev/accel/accel0"
    drv_fd = open(drv_path, "r+")
    npu_device_path = find_npu_device()
    npu_device = read_vbnv(npu_device_path)

    if args.npu_device:
        print(npu_device)
    if args.num_rows:
        print(get_core_n_rows(drv_fd))
    if args.num_cols:
        print(get_core_n_cols(drv_fd, npu_device))
    if args.aie_metadata:
        print(get_aie_metadata(drv_fd))
    if args.aie_version:
        print(get_aie_version(drv_fd))
