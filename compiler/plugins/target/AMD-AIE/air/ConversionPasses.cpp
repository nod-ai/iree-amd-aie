// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Conversion/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "air/Conversion/Passes.h.inc"
}  // namespace

namespace mlir::iree_compiler::AMDAIE {
void registerAIRConversionPasses() {
  registerAIRToAIE();
  registerAIRLoweringPass();
  registerAIRRtToNpuPass();
  registerCopyToDma();
  registerDmaToChannel();
  registerParallelToHerd();
  registerParallelToLaunch();
}
}  // namespace mlir::iree_compiler::AMDAIE
