// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Transform/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "air/Transform/Passes.h.inc"
}  // namespace

namespace mlir::iree_compiler::AMDAIE {
void registerAIRTransformPasses() {
  registerAIRDependency();
  registerAIRDependencyCanonicalize();
  registerAIRDependencyScheduleOpt();
  registerAIRFuseChannels();
  registerAIRIsolateAsyncDmaLoopNests();
  registerAIRLabelScfForLoopInAIRSegmentPattern();
  registerAIRLabelScfForLoopForPingPongPattern();
  registerAIRPingPongTransformationPattern();
  registerAIRRenumberDmaIdPass();
  registerAIRHerdPlacementPass();
  registerAIRSpecializeChannelWrapAndStridePattern();
  registerAIRSpecializeDmaBroadcast();
  registerAIRUnrollLoopForPipeliningPattern();
  registerAIRCollapseHerdPass();
  registerAIRUnrollOuterPerfectlyNestedLoopsPass();
  registerAffineLoopOptPass();
  registerAIRSplitL2MemrefForBufferConstraintPass();
  registerDmaToChannel();
  registerAIRFuseAllocDealloc();
  registerAIRShrinkMemrefSizesByAccess();
  registerAIROptimizeMemtileDMABDs();
}
}  // namespace mlir::iree_compiler::AMDAIE
