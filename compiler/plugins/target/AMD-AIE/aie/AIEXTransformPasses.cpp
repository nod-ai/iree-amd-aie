// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
}  // namespace

namespace mlir::iree_compiler::AMDAIE {
void registerAIEXTransformPasses() {
  registerAIEXToStandard();
  registerAIEDmaToNpu();
}
}  // namespace mlir::iree_compiler::AMDAIE
