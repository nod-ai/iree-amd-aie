// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEAssignBufferAddressesBasic.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
}  // namespace

namespace mlir::iree_compiler::AMDAIE {
void registerAIETransformPasses() {
  xilinx::AIE::registerAIEAssignBufferAddressesBasic();
  registerAIEAssignBufferDescriptorIDs();
  registerAIEAssignLockIDs();
  registerAIECanonicalizeDevice();
  registerAIECoreToStandard();
  registerAIELocalizeLocks();
  registerAIEObjectFifoStatefulTransform();
  registerAIERoutePathfinderFlows();
}
}  // namespace mlir::iree_compiler::AMDAIE
