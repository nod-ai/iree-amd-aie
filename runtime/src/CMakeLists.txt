# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_setup_c_src_root(
  DEFAULT_EXPORT_SET Runtime
  DEFAULT_INSTALL_COMPONENT IREEDevLibraries-Runtime
  IMPLICIT_DEFS_TARGET iree_AMD-AIE_runtime_defs
)

add_subdirectory(iree-amd-aie)
