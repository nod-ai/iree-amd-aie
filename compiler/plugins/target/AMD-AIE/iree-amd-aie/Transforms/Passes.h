// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSES_H_
#define IREE_AMD_AIE_TRANSFORMS_PASSES_H_

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
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

/// Populates passes needed to lower the IR via a Pad based approach.
void addPadBasedPassPipeline(OpPassManager &passManager,
                             TilingConfig &tilingConfig);

/// Populates passes needed to lower the IR via a Pack based approach.
void addPackBasedPassPipeline(OpPassManager &passManager,
                              TilingConfig &tilingConfig);

/// Populates passes needed to lower the IR via a simple Pack based approach.
void addSimplePackBasedPassPipeline(OpPassManager &passManager,
                                    TilingConfig &tilingConfig);

/// Create a pass to do some rewrites that help bridging the path to AIR/AIE
/// lowering.
std::unique_ptr<Pass> createAMDAIEBridgeToAIRPass();

/// Pass to bufferizes the targeted operation and materializes the result in a
/// new allocation.
std::unique_ptr<Pass> createAMDAIEBufferizeToAllocationPass(
    AMDAIEBufferizeToAllocationOptions options = {});

/// Create pass to invoke several cleanup and canonicalization patterns.
std::unique_ptr<Pass> createAMDAIECleanupPass();

/// Create a pass decomposing iree_linalg_ext.pack and unpack ops to AIR
/// dialect.
std::unique_ptr<Pass> createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass();

/// Create a pass to fuse the linalg.fill into the forall loops.
std::unique_ptr<Pass> createAMDAIEFuseFillIntoForallPass();

/// Create a pass to fuse the pack operations into the for loops.
std::unique_ptr<Pass> createAMDAIEFusePackIntoForLoopPass();

/// Create pass calling the dynamic pipeline for AMDAIE.
std::unique_ptr<Pass> createAMDAIELowerExecutableTargetPass(
    AMDAIELowerExecutableTargetOptions options = {});

/// Create pass for adding lowering strategy configurations.
std::unique_ptr<Pass> createAMDAIELoweringStrategyPass(
    AMDAIELoweringStrategyOptions options = {});

/// Create a pass to lower workgroup count region of entry point operations.
std::unique_ptr<Pass> createAMDAIELowerWorkgroupCountPass();

/// Create a pass to pack and transpose the linalg op.
std::unique_ptr<Pass> createAMDAIEPackAndTransposePass(
    AMDAIEPackAndTransposeOptions options = {});

/// Create a pass to pad MatmulOp.
std::unique_ptr<Pass> createAMDAIEPadPass(AMDAIEPadOptions options = {});

/// Create a pass to peel the first iteration out of the scf.for loop.
std::unique_ptr<Pass> createAMDAIEPeelForLoopPass();

/// Create pass to tile and fuse TilingInterface operations.
std::unique_ptr<Pass> createAMDAIETileAndFusePass(
    AMDAIETileAndFuseOptions options = {});

/// Create pass to propagate pack/unpack ops using upstream patterns.
std::unique_ptr<Pass> createAMDAIEPropagateDataLayoutPass();

void registerAMDAIEPasses();

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_PASSES_H_
