// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSES_H_
#define IREE_AMD_AIE_TRANSFORMS_PASSES_H_

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::AMDAIE {

/// Add passes to lower to AIE objectFifos.
void addAMDAIEObjectFifoLoweringPasses(OpPassManager &passManager);

/// Add passes to lower from MLIR-AIR through AIE. This is
/// currently the default passes used for lowering after IREEs tiling.
void addMLIRAIRLoweringPasses(OpPassManager &passManager, AMDAIEDevice device);

/// Add lowering passes from MLIR-AIE. This is
/// currently the default passes used for lowering from AIE dialect.
void addMLIRAIELoweringPasses(OpPassManager &passManager);

/// Populates passes needed to lower linalg/arith/math ops to LLVM dialect via
/// the structured ops path. The pass manager `pm` here operate on the module
/// within the IREE::HAL::ExecutableOp.
void buildAMDAIETransformPassPipeline(OpPassManager &pm, AMDAIEDevice device);

void buildAMDAIELowerObjectFIFO(OpPassManager &variantPassManager);

void addLowerToLLVMPasses(OpPassManager &pm);

/// Populates passes needed to lower the IR via a Pack-Peel based approach.
void addPackPeelBasedPassPipeline(OpPassManager &passManager,
                                  TilingConfig &tilingConfig);

/// Populates passes needed to lower the IR via a Pad-Pack based approach.
void addPadPackBasedPassPipeline(OpPassManager &passManager,
                                 TilingConfig &tilingConfig);

/// Populates passes needed to lower the IR via a Conv-Decompose based approach.
void addConvDecomposePassPipeline(OpPassManager &passManager,
                                  TilingConfig &tilingConfig);

/// Populates passes needed to link HAL executables across AIE targets.
void buildAMDAIELinkingPassPipeline(OpPassManager &passManager);

/// Pass to convert logical objectFifo access operations to acquire/release
/// semaphore operations.
std::unique_ptr<Pass> createAMDAIEAccessToAcquireReleasePass();

/// Create a pass to assign channels to connections.
std::unique_ptr<Pass> createAMDAIEAssignChannelsPass();

/// Create a pass to assign a buffer depth to
/// `amdaie.logicalobjectfifo.from_memref` ops.
std::unique_ptr<Pass> createAMDAIEAssignLogicalObjectFifoDepthPass(
    AMDAIEAssignLogicalObjectFifoDepthOptions options = {});

/// Create a pass to assign BD ids to `amdaie.npu.dma_cpy_nd` operations.
std::unique_ptr<Pass> createAMDAIEAssignNpuDmaBdIdsPass();

/// Create a pass to do some rewrites that help bridging the path to AIR/AIE
/// lowering.
std::unique_ptr<Pass> createAMDAIEBridgeToAIRPass();

/// Pass to bufferize the targeted operation and materialize the result in a
/// new allocation.
std::unique_ptr<Pass> createAMDAIEBufferizeToAllocationPass(
    AMDAIEBufferizeToAllocationOptions options = {});

/// Create pass to canonicalize `amdaie.npu.dma_cpy_nd` operations.
std::unique_ptr<Pass> createAMDAIECanonicalizeNpuDmaCpyNdPass();

/// Create pass to canonicalize doubly strided operations.
std::unique_ptr<Pass> createAMDAIECanonicalizeDoublyStridedOpPass(
    AMDAIECanonicalizeDoublyStridedOpOptions options = {});

/// Pass to unroll the loops within the control code regions.
std::unique_ptr<Pass> createAMDAIEControlCodeLoopUnrollPass();

/// Pass to convert `scf.forall` to `scf.for` within `aie.core`.
std::unique_ptr<Pass> createAMDAIEConvertCoreForallToForPass();

/// Pass to create a single AIE workgroup.
std::unique_ptr<Pass> createAMDAIECreateAIEWorkgroupPass();

/// Pass to create logical objectFifo link operations, explicitly linking inputs
/// and outputs.
std::unique_ptr<Pass> createAMDAIECreateLogicalObjectFifoLinkPass();

/// Pass to create references to allocations in L1 memory space.
std::unique_ptr<Pass> createAMDAIECreateReferenceToAllocationPass();

/// Create a pass to vectorize operations.
std::unique_ptr<Pass> createAMDAIEVectorizationPass();

/// Create pass to invoke several cleanup and canonicalization patterns.
std::unique_ptr<Pass> createAMDAIECleanupPass();

/// Create pass to combine strided ops within the same block if access patterns
/// are compatible.
std::unique_ptr<Pass> createAMDAIECombineStridedOpsPass();

/// Create a pass to unroll `scf.for` with synchronization ops based on
/// objectFifo buffer depths.
std::unique_ptr<Pass> createAMDAIECoreLoopUnrollPass();

/// Create a pass decomposing iree_linalg_ext.pack and unpack ops to AIR
/// dialect.
std::unique_ptr<Pass> createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass();

/// Create pass to unroll the scf.forall operations around `amdaie.core`
/// operations and distribute the logical objectFifos.
std::unique_ptr<Pass> createAMDAIEDistributeCoresAndObjectFifosPass();

/// Create a pass to compose more complex DMA operations, e.g. by combining DMA
/// operations and/or subsuming loop iterations into the strided access
/// patterns.
std::unique_ptr<Pass> createAMDAIEDmaCompositionPass(
    AMDAIEDmaCompositionOptions options = {});

/// Create a pass for common sub-expression elimination for AMDAIE DMA ops.
std::unique_ptr<Pass> createAMDAIEDmaCSEPass();

/// Create a pass to subsume loop iterations into DMA operations' access
/// patterns.
std::unique_ptr<Pass> createAMDAIEDmaLoopSubsumptionPass(
    AMDAIEDmaLoopSubsumptionOptions options = {});

/// Create a pass to convert dma operations to circular dma operations.
std::unique_ptr<Pass> createAMDAIEDmaToCircularDmaPass();

/// Create a pass to flatten the logical objectFifos.
std::unique_ptr<Pass> createAMDAIEFlattenLogicalObjectFifoPass();

/// Create a pass to fuse the consumer op into the innermost last scf loop.
std::unique_ptr<Pass> createAMDAIEFuseConsumerIntoLoopPass(
    AMDAIEFuseConsumerIntoLoopOptions options = {});

/// Create a pass to fuse the linalg.fill into the forall loops.
std::unique_ptr<Pass> createAMDAIEFuseFillIntoForallPass();

/// Hoist an affine.apply op on a scf.for op's induction variable.
std::unique_ptr<Pass> createAMDAIEHoistForLoopAffineApplyPass();

/// Create a pass to hoist logical objectFifo operations to the scope of its
/// operands.
std::unique_ptr<Pass> createAMDAIEHoistLogicalObjFifoPass();

/// Create a pass to transform linalg.generics into a form which benefits later
/// vectorization passes (to vector and aievec dialects).
std::unique_ptr<Pass> createAMDAIEInsertLoopsForVectorizationPass();

/// Create a pass to fuse the pack operations into the for loops.
std::unique_ptr<Pass> createAMDAIEFusePackIntoLoopPass(
    AMDAIEFusePackIntoLoopOptions options = {});

/// Create pass to insert `amdaie.core` operations inside the innermost
/// `scf.forall` operations selected for parallel execution.
std::unique_ptr<Pass> createAMDAIEInsertCoresPass();

/// Links AMDAIE HAL executables within the top-level program module.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createAMDAIELinkExecutablesPass();

/// Create a pass to localize logical objectfifos to local parallel loop scopes.
std::unique_ptr<Pass> createAMDAIELocalizeLogicalObjectFifoPass();

/// Create pass calling the dynamic pipeline for AMDAIE.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAMDAIELowerExecutableTargetPass(
    AMDAIELowerExecutableTargetOptions options = {});

/// Create pass for adding lowering strategy configurations.
std::unique_ptr<OperationPass<ModuleOp>> createAMDAIELoweringStrategyPass(
    AMDAIELoweringStrategyOptions options = {});

/// Create pass to lower function arguments.
std::unique_ptr<Pass> createAMDAIELowerFuncArgsPass();

/// Create pass to lower from the AMDAIE dialect to the AIE/AIEX dialects.
void addAMDAIEToAIEPasses(OpPassManager &);
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

/// Create a pass to insert a temporary buffer and replace the `None` type
/// logical objectFifo access op usage.
std::unique_ptr<Pass> createAMDAIENoneAccessToTemporaryBufferPass();

/// Normalize the loop bounds of `scf.for` and `scf.forall`.
std::unique_ptr<Pass> createAMDAIENormalizeLoopBoundsPass();

/// Create a pass to pack and transpose the linalg op.
std::unique_ptr<Pass> createAMDAIEPackAndTransposePass(
    AMDAIEPackAndTransposeOptions options = {});

/// Create pass to lower pack/unpack ops to AMDAIE DMA ops operating on 
/// logical objectFifos.
std::unique_ptr<Pass> createAMDAIEPackToDmaPass();

/// Create a pass to pad MatmulOp.
std::unique_ptr<Pass> createAMDAIEPadPass(AMDAIEPadOptions options = {});

/// Create a pass to peel the first iteration out of the scf.for loop.
std::unique_ptr<Pass> createAMDAIEPeelForLoopPass(
    AMDAIEPeelForLoopOptions options = {});

/// Create a pass to sink all dependencies into `amdaie.core` operations.
std::unique_ptr<Pass> createAMDAIESinkIntoCorePass();

/// Create a pass to split logicalobjectfifos for connection reuse.
std::unique_ptr<Pass> createAMDAIESplitLogicalObjFifosForConnectionReusePass();

/// Create pass to tile TilingInterface operations.
std::unique_ptr<Pass> createAMDAIETilePass(AMDAIETileOptions options = {});

/// Create pass to tile and fuse TilingInterface operations.
std::unique_ptr<Pass> createAMDAIETileAndFusePass(
    AMDAIETileAndFuseOptions options = {});

/// Create pass to propagate pack/unpack ops using upstream patterns.
std::unique_ptr<Pass> createAMDAIEPropagateDataLayoutPass();

void registerAMDAIEPasses();

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_PASSES_H_
