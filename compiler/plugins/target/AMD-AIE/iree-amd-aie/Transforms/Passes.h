// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSES_H_
#define IREE_AMD_AIE_TRANSFORMS_PASSES_H_

#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::AMDAIE {

/// Dummy pass that does nothing
std::unique_ptr<Pass> createPlaceholderPass();

/// Add passes to run the strategy specified using transform dialect
/// file/library
void addTransformDialectPasses(OpPassManager &passManager);

/// Populates passes needed to lower linalg/arith/math ops to LLVM dialect via
/// the structured ops path. The pass manager `pm` here operate on the module
/// within the IREE::HAL::ExecutableOp.
void buildAMDAIETransformPassPipeline(OpPassManager &pm);

/// Default pass pipeline on AMDAIE.
void addAMDAIEDefaultPassPipeline(OpPassManager &pm);

/// Create pass calling the dynamic pipeline for AMDAIE.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIELowerExecutableTargetPass();

void registerAMDAIEPasses();

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_PASSES_H_