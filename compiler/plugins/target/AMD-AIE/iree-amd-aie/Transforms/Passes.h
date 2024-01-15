// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSES_H_
#define IREE_AMD_AIE_TRANSFORMS_PASSES_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::AMDAIE {

/// Add passes to lower from MLIR-AIR through AIE. This is
/// currently the default passes used for lowering after IREEs tiling.
void addMLIRAIRAIELoweringPasses(OpPassManager &passManager);

/// Add passes to run the strategy specified using transform dialect
/// file/library
void addTransformDialectPasses(OpPassManager &passManager);

/// Populates passes needed to lower linalg/arith/math ops to LLVM dialect via
/// the structured ops path. The pass manager `pm` here operate on the module
/// within the IREE::HAL::ExecutableOp.
void buildAMDAIETransformPassPipeline(OpPassManager &pm);

/// Create a pass to do some rewrites that help bridging the path to AIR/AIE
/// lowering.
std::unique_ptr<OperationPass<>> createAMDAIEBridgeToAIRPass();

/// Create pass calling the dynamic pipeline for AMDAIE.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIELowerExecutableTargetPass();

/// Create pass to tile and fuse TilingInterface operations.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIETileAndFusePass(int64_t tilingLevel = -1);

/// Create a pass to pad MatmulOp and bufferize its operands.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIEPadAndBufferizePass(int64_t paddingLevel = -1);

/// Create pass to invoke several cleanup and canonicalization patterns.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createCleanupPass();

/// Create a pass to pack and transpose the linalg op.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIEPackAndTransposePass(int64_t packLevel = 1);

/// Pass to bufferizes the targeted operation and materializes the result in a
/// new allocation.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIEBufferizeToAllocationPass(int64_t memorySpace = 1);

void registerAMDAIEPasses();

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_PASSES_H_
