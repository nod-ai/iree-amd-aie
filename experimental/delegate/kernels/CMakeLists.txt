# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR
   NOT IREE_HAL_DRIVER_LOCAL_SYNC OR
   NOT IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  return()
endif()

message(STATUS "Processing AIE Delegate Kernels")

# Set the source and destination directories
set(_aie_delegate_kernel_src_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(_aie_delegate_kernel_dest_dir "${IREE_COMPILER_DYLIB_DIR}/iree_platform_libs/aie-delegate")

# Get a list of all files in the source directory
file(GLOB _aie_delegate_kernel_src_files "${_aie_delegate_kernel_src_dir}/*.xclbin" "${_aie_delegate_kernel_src_dir}/*.insts.txt")

message(STATUS "  PROJ BIN DIR: ${PROJECT_BINARY_DIR}")
message(STATUS "  LIB OUT DIR: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "  SRC DIR: ${_aie_delegate_kernel_src_dir}")
message(STATUS "  DEST DIR: ${_aie_delegate_kernel_dest_dir}")
message(STATUS "  SRC FILES: ${_aie_delegate_kernel_src_files}")

# Install each file to the destination directory and build a list of destination files
set(_aie_delegate_kernel_dest_files)
foreach(file ${_aie_delegate_kernel_src_files})
    message(STATUS "  Processing AIE Delegate Kernel ${file}")
    # install(FILES ${file} DESTINATION ${_aie_delegate_kernel_dest_dir})
    file(RELATIVE_PATH _relative_path ${_aie_delegate_kernel_src_dir} ${file})
    list(APPEND _aie_delegate_kernel_dest_files "${_aie_delegate_kernel_dest_dir}/${_relative_path}")
endforeach()

add_custom_command(
    OUTPUT ${_aie_delegate_kernel_dest_files}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${_aie_delegate_kernel_src_files}
        ${_aie_delegate_kernel_dest_dir}
    COMMENT "Copying AIE Delegate Kernel files"
)

message(STATUS "  DEST FILES: ${_aie_delegate_kernel_dest_files}")
add_custom_target(aie_delegate_kernels
  DEPENDS
    ${_aie_delegate_kernel_dest_files}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Ensure that the device libs are built when the compiler dylib is built.
set_property(GLOBAL APPEND PROPERTY IREE_COMPILER_DYLIB_DEPENDS
aie_delegate_kernels)
