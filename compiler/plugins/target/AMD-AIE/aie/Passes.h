// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_PASSES_H_
#define AIE_PASSES_H_

#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

namespace mlir::iree_compiler::AMDAIE {

/// Registration for AIE Transform passes.
void registerAIETransformPasses();

/// Registration for AIE Transform passes.
void registerAIEXTransformPasses();

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // AIE_PASSES_H_
