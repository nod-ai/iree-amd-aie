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

# List of all kernels stored in Azure (without file extension)
set(AIE_DELEGATE_KERNELS
    "matmul/matmul-bf16-256x256x256-v1"
)

# List of all kernel file extensions
set(AIE_DELEGATE_KERNEL_EXTENSIONS
    ".insts.txt"
    ".xclbin"
)

# Kernel repo
set(AIE_DELEGATE_KERNEL_REPO
    "https://sharkpublic.blob.core.windows.net/sharkpublic/aie-delegate/kernels"
)

# Install each file to the destination directory and build a list of destination files
set(AIE_DELEGATE_KERNEL_DEST_FILES)
foreach(kernel ${AIE_DELEGATE_KERNELS})
    foreach (extension ${AIE_DELEGATE_KERNEL_EXTENSIONS})
        set(_src_file "${kernel}${extension}")
        set(_dest_file "${CMAKE_CURRENT_BINARY_DIR}/${_src_file}")
        set(_url "${AIE_DELEGATE_KERNEL_REPO}/${_src_file}")
        add_custom_command(
            OUTPUT ${_dest_file}
            COMMAND wget -O ${_dest_file} ${_url}
            COMMENT "wget ${_url}"
        )

        list(APPEND AIE_DELEGATE_KERNEL_DEST_FILES "${_dest_file}")
        unset(_url)
        unset(_dest_file)
        unset(_src_file)
    endforeach()
endforeach()

add_custom_target(aie_delegate_kernels
  DEPENDS
    ${AIE_DELEGATE_KERNEL_DEST_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
