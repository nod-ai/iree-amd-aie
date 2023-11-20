// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIR_PASSES_H_
#define AIR_PASSES_H_

namespace mlir::iree_compiler::AMDAIE {

/// Registration for AIR Conversion passes.
void registerAIRConversionPasses();

/// Registration for AIR Transform passes.
void registerAIRTransformPasses();

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // AIR_PASSES_H_
