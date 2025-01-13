#!/bin/bash
#
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -e

sudo modprobe -r amdxdna
sudo modprobe drm_shmem_helper
sudo modprobe amdxdna dyndbg==pflm timeout_in_sec=10
