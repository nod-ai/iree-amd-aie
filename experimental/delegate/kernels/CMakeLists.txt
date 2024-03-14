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

# Get a list of all files in the source directory
file(GLOB _aie_delegate_kernel_src_files "${CMAKE_CURRENT_SOURCE_DIR}/*.xclbin" "${CMAKE_CURRENT_SOURCE_DIR}/*.insts.txt")

message(STATUS "  PROJ BIN DIR: ${PROJECT_BINARY_DIR}")
message(STATUS "  SRC DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  DEST DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "  SRC FILES: ${_aie_delegate_kernel_src_files}")

# Install each file to the destination directory and build a list of destination files
set(_aie_delegate_kernel_dest_files)
foreach(file ${_aie_delegate_kernel_src_files})
    message(STATUS "  Processing AIE Delegate Kernel ${file}")
    # install(FILES ${file} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(RELATIVE_PATH _relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${file})
    list(APPEND _aie_delegate_kernel_dest_files "${CMAKE_CURRENT_BINARY_DIR}/${_relative_path}")
endforeach()

add_custom_command(
    OUTPUT ${_aie_delegate_kernel_dest_files}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${_aie_delegate_kernel_src_files}
        ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Copying AIE Delegate Kernel files"
)

message(STATUS "  DEST FILES: ${_aie_delegate_kernel_dest_files}")
add_custom_target(aie_delegate_kernels
  DEPENDS
    ${_aie_delegate_kernel_dest_files}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
