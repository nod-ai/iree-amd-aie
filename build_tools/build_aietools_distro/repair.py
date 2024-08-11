# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import stat
from os.path import isabs, abspath, basename, exists
from os.path import join as pjoin
from pathlib import Path

from auditwheel.policy import WheelPolicies
from auditwheel.wheel_abi import get_wheel_elfdata
from auditwheel.wheeltools import InWheelCtx


def repair_wheel(
    wheel_policy,
    wheel_path: str,
    abis: list[str],
    dest_dir: str,
    out_dir: str,
    exclude: frozenset[str],
) -> str | None:
    external_refs_by_fn = get_wheel_elfdata(wheel_policy, wheel_path, exclude)[1]
    if not isabs(out_dir):
        out_dir = abspath(out_dir)

    wheel_fname = basename(wheel_path)

    with InWheelCtx(wheel_path) as ctx:
        ctx.out_wheel = pjoin(out_dir, wheel_fname)
        if not exists(dest_dir):
            os.mkdir(dest_dir)

        # here, fn is a path to an ELF file (lib or executable) in
        # the wheel, and v['libs'] contains its required libs
        for fn, v in external_refs_by_fn.items():
            ext_libs: dict[str, str] = v[abis[0]]["libs"]
            for soname, src_path in ext_libs.items():
                assert soname not in exclude

                if src_path is None:
                    raise ValueError(soname)

                copylib(src_path, dest_dir)

    return ctx.out_wheel


def copylib(src_path: str, dest_dir: str) -> tuple[str, str]:
    src_name = os.path.basename(src_path)
    new_soname = src_name

    dest_path = os.path.join(dest_dir, new_soname)
    if os.path.exists(dest_path):
        return new_soname, dest_path

    shutil.copy2(src_path, dest_path)
    statinfo = os.stat(dest_path)
    if not statinfo.st_mode & stat.S_IWRITE:
        os.chmod(dest_path, statinfo.st_mode | stat.S_IWRITE)

    return new_soname, dest_path


wheel_policy = WheelPolicies()
out_wheel = repair_wheel(
    wheel_policy,
    str(Path.cwd() / "dist" / "chess-0.0.0-py3-none-any.whl"),
    abis=["manylinux_2_35_x86_64"],
    dest_dir="Vitis/2024.1/aietools/lib/lnx64.o",
    out_dir=str(Path.cwd() / "dist"),
    exclude=frozenset(),
)
