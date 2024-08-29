# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_AMD_AIE_ENABLE_XRT_DRIVER)
    add_subdirectory(driver/xrt)
endif()

if(IREE_AMD_AIE_ENABLE_HSA_DRIVER)
    add_subdirectory(driver/hsa)
endif()

# Flatbuffer schema generation does not require XRT. Moreover the generated
# flatbuffer header files are used by the compiler to create artefacts
# (.vmfb file), and so the schema sub-directory is required even when not
# building the XRT driver code.
add_subdirectory(schemas)

# Contains libiree_aie_runtime, i.e., suitably encapsulated calls to aie-rt.
# Note, "runtime" is currently a misnomer as all such calls are made at
# compile/configuration time; the name is inspired by aie-rt itself
# (where rt stands for runtime) and the fact that in the future, the plan is indeed
# to make these same calls at runtime.
add_subdirectory(aie_runtime)
