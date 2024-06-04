# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_AMD_AIE_ENABLE_XRT_DRIVER)
    add_subdirectory(driver/xrt)
endif()

# Flatbuffer schema generation does not require XRT. Moreover the generated
# flatbuffer header files are used by the compiler to create artefacts
# (.vmfb file), and so the schema sub-directory is required even when not
# building the XRT driver code.
add_subdirectory(schemas)

add_subdirectory(runtime)
