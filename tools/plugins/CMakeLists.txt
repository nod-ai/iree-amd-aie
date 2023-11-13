# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/tools")
iree_add_all_subdirs()
set(IREE_PACKAGE_ROOT_PREFIX "")

# If a target backend that requires LLD to link codegen executables is
# enabled, install the target.
if(IREE_LLD_TARGET)
  install(
    TARGETS lld
    COMPONENT Compiler
    RUNTIME DESTINATION bin
  )
endif()

iree_cc_binary(
  NAME
    iree-aie-translate
  SRCS
    "iree-aie-translate.cpp"
  DEPS
    iree::compiler::Dialect::HAL::IR::HALDialect
    iree::compiler::Dialect::Stream::IR
    iree::target::amd-aie::Translation::AIESerializer
    MLIRGPUDialect
    MLIRSCFDialect
    MLIRTranslateLib
    MLIRSupport
    IREELinalgExtDialect
  DATA
    ${IREE_LLD_TARGET}
  HOSTONLY
  SETUP_INSTALL_RPATH
)
