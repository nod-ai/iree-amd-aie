# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    Target
  HDRS
    "AIETarget.h"
    "XclBinGeneratorKit.h"
  SRCS
    "AIETarget.cpp"
    "XclBinGeneratorKit.cpp"
  DEPS
    iree::compiler::Dialect::HAL::Target
    iree::target::amd-aie::Transforms
    iree::target::amd-aie::aie::AIETranslationPasses
    iree::target::amd-aie::aie::AIEXTransformPasses
    iree::target::amd-aie::air::AIRDialectIR
  PUBLIC
)

# Set up copy rules.
set(_platform_lib_reldir "iree_platform_libs/amd-aie")
set(_platform_lib_absdir "${IREE_COMPILER_DYLIB_DIR}/${_platform_lib_reldir}")
file(MAKE_DIRECTORY "${_platform_lib_absdir}")
set(_all_device_bc_copy_commands)
set(_all_device_bc_files)

# Copy to lib/ tree.
set(_device_bc_srcpath "${IREE_MLIR_AIE_SOURCE_DIR}/aie_runtime_lib/AIE2/me_basic.o")
set(_device_bc_relpath "${_platform_lib_reldir}/me_basic.o")
list(APPEND _all_device_bc_files "${IREE_COMPILER_DYLIB_DIR}/${_device_bc_relpath}")
list(APPEND _all_device_bc_deps "${_device_bc_path}")
list(APPEND _all_device_bc_copy_commands
  COMMAND ${CMAKE_COMMAND} -E copy
    "${_device_bc_srcpath}"
    "${IREE_COMPILER_DYLIB_DIR}/${_device_bc_relpath}"
)

# Note this bc file as being part of the bundle that must be included with
# the compiler dylib.
set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_DYLIB_RELPATHS "${_device_bc_relpath}")

# Generate a custom target with all file level dependencies and commands to
# copy to our build tree locations.
# Our GenDeviceLibs target depends on all of the defined device lib targets.
add_custom_command(
  OUTPUT ${_all_device_bc_files}
  DEPENDS ${_all_device_bc_deps}
  POST_BUILD
    ${_all_device_bc_copy_commands}
)
add_custom_target(iree_compiler_plugins_target_AMDAIE_GenDeviceLibs
  DEPENDS
    ${_all_device_bc_files}
)

# Ensure that the device libs are built when the compiler dylib is built.
set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_DYLIB_DEPENDS
  iree_compiler_plugins_target_AMDAIE_GenDeviceLibs)

# Install.
install(FILES ${_all_device_bc_files}
  DESTINATION "${IREE_COMPILER_DYLIB_INSTALL_PREFIX}/${_platform_lib_reldir}")

set(VITIS_ROOT /proj/xbuilds/2023.2_released/installs/lin64/Vitis/2023.2)
set(VITIS_AIETOOLS_DIR /proj/xbuilds/2023.2_released/installs/lin64/Vitis/2023.2/aietools)
set(AIE_RUNTIME_TARGETS x86_64)
add_subdirectory(${IREE_MLIR_AIE_SOURCE_DIR}/runtime_lib runtime_lib)
add_subdirectory(${IREE_MLIR_AIE_SOURCE_DIR}/data data)