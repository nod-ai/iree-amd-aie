// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
}  // namespace

namespace mlir::iree_compiler::AMDAIE {
void registerAIETransformPasses() {
  registerAIEAssignLockIDs();
  registerAIEAssignBufferAddresses();
  registerAIECanonicalizeDevice();
  registerAIEObjectFifoRegisterProcess();
  registerAIEObjectFifoStatefulTransform();
  registerAIERoutePacketFlows();
}
}  // namespace mlir::iree_compiler::AMDAIE
