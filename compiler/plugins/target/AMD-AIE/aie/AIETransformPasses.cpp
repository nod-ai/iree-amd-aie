// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEAssignBufferAddressesBasic.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
}  // namespace

namespace mlir::iree_compiler::AMDAIE {
void registerAIETransformPasses() {
  registerAIEAssignLockIDs();
  registerAIEAssignBufferDescriptorIDs();
  xilinx::AIE::registerAIEAssignBufferAddressesBasic();
  registerAIECanonicalizeDevice();
  registerAIECoreToStandard();
  registerAIERoutePathfinderFlows();
  registerAIELocalizeLocks();
  registerAIENormalizeAddressSpaces();
  registerAIEObjectFifoRegisterProcess();
  registerAIEObjectFifoStatefulTransform();
  registerAIERoutePacketFlows();
  // convert to llvm
  DialectRegistry registry;
  registerAllToLLVMIRTranslations(registry);
  arith::registerConvertArithToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  func::registerAllExtensions(registry);
  registerConvertFuncToLLVMInterface(registry);
  index::registerConvertIndexToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
}
}  // namespace mlir::iree_compiler::AMDAIE
