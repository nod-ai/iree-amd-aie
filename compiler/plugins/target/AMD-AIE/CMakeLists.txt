# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_directories("${IREE_MLIR_AIE_SOURCE_DIR}/include")
include_directories("${IREE_MLIR_AIR_SOURCE_DIR}/include")

iree_setup_c_src_root(
  PACKAGE_ROOT_PREFIX iree::target
  IMPLICIT_DEFS_TARGET iree_target_amd-aie_compiler_defs
)

add_subdirectory(iree-amd-aie)
add_subdirectory(aie)
add_subdirectory(aievec)
add_subdirectory(air)
