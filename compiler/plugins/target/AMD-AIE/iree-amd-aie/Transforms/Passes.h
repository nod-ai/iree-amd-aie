// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSES_H_
#define IREE_AMD_AIE_TRANSFORMS_PASSES_H_

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::AMDAIE {

/// Add passes to lower from MLIR-AIR through AIE. This is
/// currently the default passes used for lowering after IREEs tiling.
void addMLIRAIRAIELoweringPasses(OpPassManager &passManager, bool packPeel);

/// Populates passes needed to lower linalg/arith/math ops to LLVM dialect via
/// the structured ops path. The pass manager `pm` here operate on the module
/// within the IREE::HAL::ExecutableOp.
void buildAMDAIETransformPassPipeline(OpPassManager &pm);

/// Populates passes needed to lower the IR via a Pack based approach.
void addPackPeelBasedPassPipeline(OpPassManager &passManager,
                                  TilingConfig &tilingConfig);

/// Populates passes needed to lower the IR via a Pad-Pack based approach.
void addPadPackBasedPassPipeline(OpPassManager &passManager,
                                 TilingConfig &tilingConfig);

/// Create a pass to convert AIR DMA ops into AMDAIE DMA ops operating on
/// logical objectfifos.
std::unique_ptr<Pass> createAMDAIEAIRDmaAMDAIEDmaPass();

/// Create a pass to do some rewrites that help bridging the path to AIR/AIE
/// lowering.
std::unique_ptr<Pass> createAMDAIEBridgeToAIRPass();

/// Pass to bufferizes the targeted operation and materializes the result in a
/// new allocation.
std::unique_ptr<Pass> createAMDAIEBufferizeToAllocationPass(
    AMDAIEBufferizeToAllocationOptions options = {});

/// Create pass to apply caonicaliztions to air.dma_memcpy_nd op's.
std::unique_ptr<Pass> createAMDAIECanonicalizeDmaPass();

/// Create pass to canonicalize doubly strided operations.
std::unique_ptr<Pass> createAMDAIECanonicalizeDoublyStridedOpPass();

/// Pass to convert logical objectfifo consume/produce operation to
/// acquire/release semaphore operations.
std::unique_ptr<Pass> createAMDAIEConsumeProduceToAcquireReleasePass();

/// Pass to unroll the loops within the control code regions.
std::unique_ptr<Pass> createAMDAIEControlCodeLoopUnrollPass();

/// Pass to create a single AIE workgroup.
std::unique_ptr<Pass> createAMDAIECreateAIEWorkgroupPass();

/// Pass to create logical objectfifo link operations, explicitly linking inputs
/// and outputs.
std::unique_ptr<Pass> createAMDAIECreateLogicalObjectFifoLinkPass();

/// Create a pass to vectorize operations.
std::unique_ptr<Pass> createAMDAIEVectorizationPass();

/// Create pass to invoke several cleanup and canonicalization patterns.
std::unique_ptr<Pass> createAMDAIECleanupPass();

/// Create a pass decomposing iree_linalg_ext.pack and unpack ops to AIR
/// dialect.
std::unique_ptr<Pass> createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass();

/// Create a pass to convert dma operations to circular dma operations.
std::unique_ptr<Pass> createAMDAIEDmaToCircularDmaPass();

/// Create a pass to fuse the consumer op into the innermost last scf loop.
std::unique_ptr<Pass> createAMDAIEFuseConsumerIntoLoopPass(
    AMDAIEFuseConsumerIntoLoopOptions options = {});

/// Create a pass to fuse the linalg.fill into the forall loops.
std::unique_ptr<Pass> createAMDAIEFuseFillIntoForallPass();

/// Create a pass to fuse or distribute logical objectfifos into workgroups.
std::unique_ptr<Pass> createAMDAIEFuseLogicalObjectFifoIntoWorkgroupPass();

/// Hoist an affine apply op on a scf.for op's induction variable.
std::unique_ptr<Pass> createAMDAIEHoistForLoopAffineApplyPass();

/// Create a pass to transform linalg.generics into a form which benefits later
/// vectorization passes (to vector and aievec dialects).
std::unique_ptr<Pass> createAMDAIEInsertLoopsForVectorizationPass();

/// Create a pass to fuse the pack operations into the for loops.
std::unique_ptr<Pass> createAMDAIEFusePackIntoLoopPass(
    AMDAIEFusePackIntoLoopOptions options = {});

/// Create pass to insert AIE workgroups around forall loops selected
/// for parallel execution.
std::unique_ptr<Pass> createAMDAIEInsertAIEWorkgroupPass();

/// Create pass calling the dynamic pipeline for AMDAIE.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAMDAIELowerExecutableTargetPass(
    AMDAIELowerExecutableTargetOptions options = {});

/// Create pass for adding lowering strategy configurations.
std::unique_ptr<OperationPass<ModuleOp>> createAMDAIELoweringStrategyPass(
    AMDAIELoweringStrategyOptions options = {});

/// Create pass to lower from the AMDAIE dialect to the AIE/AIEX dialects.
std::unique_ptr<Pass> createAMDAIELowerToAIEPass();

/// Create pass to lower a sequence of operation(s) to a iree_codegen.ukernel.*
/// operation.
std::unique_ptr<Pass> createAMDAIELowerToUKernelsPass(
    AMDAIELowerToUKernelsOptions options = {});

/// Create a pass to lower workgroup count region of entry point operations.
std::unique_ptr<Pass> createAMDAIELowerWorkgroupCountPass();

/// Create a pass to map scf.forall ops to blocks and cores.
std::unique_ptr<Pass> createAMDAIEMapForallToCoresPass(
    AMDAIEMapForallToCoresOptions options = {});

/// Normalize the loop bounds of `scf.for` and `scf.forall`.
std::unique_ptr<Pass> createAMDAIENormalizeLoopBoundsPass();

/// Create a pass to pack and transpose the linalg op.
std::unique_ptr<Pass> createAMDAIEPackAndTransposePass(
    AMDAIEPackAndTransposeOptions options = {});

/// Create pass to lower pack/unpack ops to air.DmaMemcpyNd ops.
std::unique_ptr<Pass> createAMDAIEPackToDmaPass();

/// Create a pass to pad MatmulOp.
std::unique_ptr<Pass> createAMDAIEPadPass(AMDAIEPadOptions options = {});

/// Create a pass to peel the first iteration out of the scf.for loop.
std::unique_ptr<Pass> createAMDAIEPeelForLoopPass(
    AMDAIEPeelForLoopOptions options = {});

/// Create pass to tile TilingInterface operations.
std::unique_ptr<Pass> createAMDAIETilePass(AMDAIETileOptions options = {});

/// Create pass to tile and fuse TilingInterface operations.
std::unique_ptr<Pass> createAMDAIETileAndFusePass(
    AMDAIETileAndFuseOptions options = {});

/// Create pass to propagate pack/unpack ops using upstream patterns.
std::unique_ptr<Pass> createAMDAIEPropagateDataLayoutPass();

/// Create pass to unroll the scf.forall operations within AIE workgroups and
/// distribute the logical objectfifos .
std::unique_ptr<Pass> createAMDAIEUnrollAndDistributeWorkgroupPass();

void registerAMDAIEPasses();

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_PASSES_H_
