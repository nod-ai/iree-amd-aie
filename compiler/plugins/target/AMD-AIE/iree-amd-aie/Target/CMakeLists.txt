# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    AMDAIERT
  HDRS
    "AMDAIERT.h"  
  SRCS
    "AMDAIERT.cpp"
  DEPS
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
)

iree_cc_library(
  NAME
    AIETargets
  SRCS
    "AMDAIETargetBCF.cpp"
    "AMDAIETargetCDODirect.cpp"
    "AMDAIETargetLdScript.cpp"
    "XCLBinGen.cpp"
  DEPS
    ::AMDAIERT
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
    iree::target::amd-aie::Transforms
    iree-aie-bootgen
)

if(IREE_AMD_AIE_ENABLE_XRT_DRIVER)
  add_dependencies(iree_target_amd-aie_Target_AIETargets iree-aie-xclbinutil)
endif()

iree_cc_library(
  NAME
    Target
  HDRS
    "AIETarget.h"
  SRCS
    "AIETarget.cpp"
  DEPS
    ::AIETargets
    ::AMDAIERT
    iree-amd-aie::schemas::xrt_executable_def_c_fbs
    iree-amd-aie::schemas::pdi_executable_def_c_fbs
    iree::base::internal::flatcc::building
    iree::base::internal::flatcc::parsing
    iree::compiler::Dialect::HAL::Target
    iree::target::amd-aie::IR::AMDAIEDialect
    iree::target::amd-aie::Transforms
    iree::target::amd-aie::air::AIRDialectIR
    MLIRLLVMDialect
  PUBLIC
)
