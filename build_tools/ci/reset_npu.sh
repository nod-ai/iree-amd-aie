#!/bin/bash
#
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -e

NUMBER=$(lspci -D | grep "\[AMD\] Device 1502" | cut -d ' ' -f1)

if [ x"$NUMBER" != x"" ]; then
  sudo modprobe -r amdxdna
  sudo modprobe drm_shmem_helper
  sudo modprobe amdxdna dyndbg==pflm
else
  echo "couldn't find npu"
fi

