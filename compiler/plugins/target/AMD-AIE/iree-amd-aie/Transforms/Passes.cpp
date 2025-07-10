// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"

#include "aie/Passes.h"
#include "aievec/Passes.h"
#include "air/Conversion/AIRLoweringPass.h"
#include "air/Conversion/AIRRtToNpuPass.h"
#include "air/Conversion/AIRToAIEPass.h"
#include "air/Conversion/ConvertToAIRPass.h"
#include "air/Transform/AIRDependency.h"
#include "air/Transform/AIRDependencyCanonicalize.h"
#include "air/Transform/AIRDependencyScheduleOpt.h"
#include "air/Transform/AIRDmaToChannel.h"
#include "air/Transform/AIRHerdPlacementPass.h"
#include "air/Transform/AIRMiscPasses.h"
#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-amdaie-lowering-pass-pipeline"

namespace mlir::iree_compiler::AMDAIE {

void appendVectorizationToPipeline(OpPassManager &funcPassManager,
                                   bool enableVectorizationPasses,
                                   bool enableCoalescingLoops = false,
                                   bool enableCollapsingUnitDims = false) {
  if (!enableVectorizationPasses) return;
  funcPassManager.addPass(createAMDAIECleanupPass());
  {
    AMDAIEInsertLoopsForVectorizationOptions options;
    options.enableCoalescing = enableCoalescingLoops;
    options.enableCollapsingUnitDims = enableCollapsingUnitDims;
    funcPassManager.addPass(
        createAMDAIEInsertLoopsForVectorizationPass(options));
  }
  funcPassManager.addPass(createAMDAIEVectorizationPass());
}

//===---------------------------------------------------------------------===//
// Default allocation functions for AIE backend
//===---------------------------------------------------------------------===//

static LogicalResult aieComprehensiveBufferizeCopyFn(OpBuilder &builder,
                                                     Location loc, Value from,
                                                     Value to) {
  // TODO: ideally we should use linalg.copy which was recently reintroduced
  // as an OpDSL named op. However, IREE-specific patterns to cleanup spurious
  // post-bufferization copies do not trigger properly.
  // So we keep using `createLinalgCopyOp` which builds a GenericOp.
  // builder.create<linalg::CopyOp>(loc, from, to);
  mlir::iree_compiler::createLinalgCopyOp(builder, loc, from, to);
  return success();
}

static void addAMDAIEBufferizePasses(OpPassManager &pm,
                                     TilePassPipeline useTilePipeline) {
  // Bufferize the dispatch.
  using mlir::bufferization::BufferizationOptions;

  // Allocation callbacks to use with upstream comprehensive bufferization
  auto aieComprehensiveBufferizeAllocationFn =
      [useTilePipeline](OpBuilder &builder, Location loc, MemRefType memRefType,
                        ValueRange dynamicSizes, unsigned _alignment) {
        int64_t numDims = memRefType.getShape().size();
        AMDAIEMemSpace memSpace = AMDAIEMemSpace::Local;
        if ((useTilePipeline == TilePassPipeline::PackPeelPipeline ||
             useTilePipeline ==
                 TilePassPipeline::PackPeel4LevelTilingPipeline) &&
            numDims == 4) {
          memSpace = AMDAIEMemSpace::Shared;
        }

        OpBuilder::InsertionGuard g(builder);
        auto memorySpaceAttr =
            AMDAIEMemSpaceAttr::get(builder.getContext(), memSpace);
        MemRefType allocType =
            MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                            AffineMap(), memorySpaceAttr);
        return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes)
            .getResult();
      };

  BufferizationOptions::AllocationFn allocationFn =
      aieComprehensiveBufferizeAllocationFn;
  BufferizationOptions::MemCpyFn memCpyFn = aieComprehensiveBufferizeCopyFn;
  addIREEComprehensiveBufferizePasses(pm, allocationFn, memCpyFn);
}

void addAMDAIEToAIEPasses(OpPassManager &passManager,
                          bool insertLoopAroundCoreBlock) {
  // The infinite loop insertion transformation needs to be called before the
  // `AcquireReleaseToUseLock` pass as the latter will perform loop unrolling
  // based on the objFifo depths.
  // TODO(jornt): Make them independent.
  if (insertLoopAroundCoreBlock)
    passManager.addPass(createAMDAIEInsertInfiniteLoopAroundCoreBlockPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createAMDAIEAcquireReleaseToUseLockPass());
  passManager.addPass(createAMDAIECanonicalizeNpuDmaCpyNdPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createAMDAIESinkIntoCorePass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createAMDAIEAddNoAliasFunctionArgumentsPass());
  passManager.addPass(createAMDAIELowerToAIEPass());
  passManager.addPass(createAMDAIERemoveMemorySpacePass());
  passManager.addPass(createCanonicalizerPass());
}

void addPeelAndFusePasses(OpPassManager &funcPassManager) {
  // Hoist static allocations
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Peel the first and last iteration. Note: Do not run CSE pass afterwards,
  // because it may bring problem for bufferization.
  funcPassManager.addPass(createAMDAIEPeelForLoopPass());
  funcPassManager.addPass(createCanonicalizerPass());

  // Fuse fill op into the first inner forall loop
  funcPassManager.addPass(createAMDAIEFuseFillIntoForallPass());

  // Fuse unpack/elementwise consumer ops into the last inner forall loop
  funcPassManager.addPass(createAMDAIEFuseConsumerIntoLoopPass());

  // Note: canonicalizer pass should not run starting from here until
  // bufferization to avoid creating redundant allocation and data copy.
  // TODO (vivian): solve the bufferization problem upstream

  // Fuse pack ops into the last inner forall loop
  {
    AMDAIEFuseProducerIntoLoopOptions fuseProducerOptions;
    fuseProducerOptions.fuseDepth = 1;
    fuseProducerOptions.useSCFFor = false;
    fuseProducerOptions.targetElementwise = true;
    funcPassManager.addPass(
        createAMDAIEFuseProducerIntoLoopPass(fuseProducerOptions));
  }

  // Promote the elementwise input to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 2;
    bufferizeOptions.bufferizeElementwise = true;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::LinalgInput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }
}

void addPackPeelBasedPassPipeline(OpPassManager &funcPassManager,
                                  TilePassPipeline useTilePipeline) {
  // First level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.hardwareMapping = HardwareMapping::Block;
    tileFuseOptions.tilingLevel = 0;
    tileFuseOptions.useSCFFor = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // First level packing
  {
    AMDAIEPackAndTransposeOptions packOptions;
    packOptions.packLevel = 0;
    funcPassManager.addPass(createAMDAIEPackAndTransposePass(packOptions));
  }

  // Propagate pack ops for the elementwise op
  funcPassManager.addPass(createAMDAIEPropagateDataLayoutPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the matmul output to shared memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 1;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::LinalgOutput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Promote the elementwise input to shared memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 1;
    bufferizeOptions.bufferizeElementwise = true;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::LinalgInput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Second level packing
  {
    AMDAIEPackAndTransposeOptions packOptions;
    packOptions.packLevel = 1;
    funcPassManager.addPass(createAMDAIEPackAndTransposePass(packOptions));
  }

  // Propagate pack ops for the elementwise op
  funcPassManager.addPass(createAMDAIEPropagateDataLayoutPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Tile the reduction dimension using scf.for
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.tilingLevel = 1;
    tileFuseOptions.useSCFFor = true;
    tileFuseOptions.tileElementwise = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Fuse both levels of pack ops into for loop
  {
    AMDAIEFuseProducerIntoLoopOptions fuseProducerOptions;
    fuseProducerOptions.fuseDepth = 2;
    fuseProducerOptions.useSCFFor = true;
    funcPassManager.addPass(
        createAMDAIEFuseProducerIntoLoopPass(fuseProducerOptions));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the operands of pack at depth 2 from the linalg ops to shared
  // memory.
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 1;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::PackOrCopyInput;
    bufferizeOptions.inputDepth = 2;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Second level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.hardwareMapping = HardwareMapping::Core;
    tileFuseOptions.tilingLevel = 2;
    tileFuseOptions.useSCFFor = false;
    tileFuseOptions.tileElementwise = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Fuse second level pack ops into forall loop
  {
    AMDAIEFuseProducerIntoLoopOptions fuseProducerOptions;
    fuseProducerOptions.fuseDepth = 1;
    fuseProducerOptions.useSCFFor = false;
    funcPassManager.addPass(
        createAMDAIEFuseProducerIntoLoopPass(fuseProducerOptions));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the matmul inputs to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 2;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::LinalgInput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Peel the for loop and fuse ops into the loops
  addPeelAndFusePasses(funcPassManager);

  // Lower to UKernels.
  funcPassManager.addPass(createAMDAIELowerToUKernelsPass());

  // Comprehensive bufferization
  addAMDAIEBufferizePasses(funcPassManager, useTilePipeline);
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
}

void addPackPeel4LevelTilingBasedPassPipeline(OpPassManager &funcPassManager,
                                              TilePassPipeline useTilePipeline,
                                              Operation *rootOp) {
  // Check if the root op is a 4D matmul-like operation.
  auto linalgRootOp = dyn_cast<linalg::LinalgOp>(rootOp);
  bool is4DMatmulOp = is4DMatmulLikeOp(linalgRootOp);

  // First level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.tilingLevel = 0;
    tileFuseOptions.useSCFFor = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Data movement from L3 to L2 memory space using pack or pad operation.
  // For 2D matmul-like ops, pack operation is used to expand operands from 2D
  // to 4D. For 4D matmul-like ops, pad operation is used to keep the original
  // dimensions.
  if (is4DMatmulOp) {
    // First level pad
    if (isMatmulWithElementwiseConsumer(linalgRootOp)) {
      // For matmul-elementwise case, first pad the output of the elementwise op
      // and then pad the inputs of matmul op.
      {
        AMDAIEPadOptions padOptions;
        padOptions.padElementwise = true;
        padOptions.padOperand = PadOperand::Output;
        funcPassManager.addPass(createAMDAIEPadPass(padOptions));
      }
      {
        AMDAIEPadOptions padOptions;
        padOptions.padElementwise = false;
        padOptions.padOperand = PadOperand::Input;
        funcPassManager.addPass(createAMDAIEPadPass(padOptions));
      }
    } else {
      // For matmul-like op, pad both inputs and output of the root operation.
      {
        AMDAIEPadOptions padOptions;
        padOptions.padElementwise = false;
        padOptions.padOperand = PadOperand::InputOutput;
        funcPassManager.addPass(createAMDAIEPadPass(padOptions));
      }
    }
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  } else {
    // First level packing
    {
      AMDAIEPackAndTransposeOptions packOptions;
      packOptions.packLevel = 0;
      funcPassManager.addPass(createAMDAIEPackAndTransposePass(packOptions));
    }
    // Propagate pack ops for the elementwise op
    funcPassManager.addPass(createAMDAIEPropagateDataLayoutPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  // Promote the matmul output to shared memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 1;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::LinalgOutput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Promote the elementwise input to shared memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 1;
    bufferizeOptions.bufferizeElementwise = true;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::LinalgInput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Data movement from L2 to L1 memory space using pack operation.
  // If the input is 4D matmul-like op, this is the first level of packing.
  // Otherwise for 2D matmul-like op, it is the second level.
  {
    AMDAIEPackAndTransposeOptions packOptions;
    packOptions.packLevel = is4DMatmulOp ? 0 : 1;
    funcPassManager.addPass(createAMDAIEPackAndTransposePass(packOptions));
  }

  // Propagate pack ops for the elementwise op
  funcPassManager.addPass(createAMDAIEPropagateDataLayoutPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Second level tiling using scf.forall.
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.tilingLevel = 1;
    tileFuseOptions.useSCFFor = false;
    tileFuseOptions.tileElementwise = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Fuse the second level of pack ops into forall loop
  {
    AMDAIEFuseProducerIntoLoopOptions fuseProducerOptions;
    fuseProducerOptions.fuseDepth = 1;
    fuseProducerOptions.useSCFFor = false;
    funcPassManager.addPass(
        createAMDAIEFuseProducerIntoLoopPass(fuseProducerOptions));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Tile the reduction dimension using scf.for
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.tilingLevel = 2;
    tileFuseOptions.useSCFFor = true;
    tileFuseOptions.tileElementwise = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Fuse the second level of pack ops into for loop
  {
    AMDAIEFuseProducerIntoLoopOptions fuseProducerOptions;
    fuseProducerOptions.fuseDepth = 1;
    fuseProducerOptions.useSCFFor = true;
    funcPassManager.addPass(
        createAMDAIEFuseProducerIntoLoopPass(fuseProducerOptions));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the operands of pack at depth 2 from the linalg ops to shared
  // memory.
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 1;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::PackOrCopyInput;
    bufferizeOptions.inputDepth = 2;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Third level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.hardwareMapping = HardwareMapping::Core;
    tileFuseOptions.tilingLevel = 3;
    tileFuseOptions.useSCFFor = false;
    tileFuseOptions.tileElementwise = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Fuse second level pack ops into forall loop
  {
    AMDAIEFuseProducerIntoLoopOptions fuseProducerOptions;
    fuseProducerOptions.fuseDepth = 1;
    fuseProducerOptions.useSCFFor = false;
    funcPassManager.addPass(
        createAMDAIEFuseProducerIntoLoopPass(fuseProducerOptions));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the matmul inputs to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 2;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::LinalgInput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Peel the for loop and fuse ops into the loops
  addPeelAndFusePasses(funcPassManager);

  // Lower to UKernels.
  funcPassManager.addPass(createAMDAIELowerToUKernelsPass());

  // Comprehensive bufferization
  addAMDAIEBufferizePasses(funcPassManager, useTilePipeline);
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
}

void addConvDecomposePassPipeline(OpPassManager &funcPassManager,
                                  TilePassPipeline useTilePipeline) {
  auto addCleanups = [&]() {
    funcPassManager.addPass(createAMDAIECleanupPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  };

  // First level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.hardwareMapping = HardwareMapping::Block;
    tileFuseOptions.tilingLevel = 0;
    tileFuseOptions.useSCFFor = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
    addCleanups();
  }

  // Pad the linalg operation
  {
    AMDAIEPadOptions padOptions;
    padOptions.padOperand = PadOperand::InputOutput;
    funcPassManager.addPass(createAMDAIEPadPass(padOptions));
  }

  // Promote the input and result to shared memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 1;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::LinalgInputOutput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Second level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.hardwareMapping = HardwareMapping::Core;
    tileFuseOptions.tilingLevel = 1;
    tileFuseOptions.useSCFFor = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
    addCleanups();
  }

  // Fuse fill op into the inner forall loop
  funcPassManager.addPass(createAMDAIEFuseFillIntoForallPass());

  // Pack the linalg operation
  {
    AMDAIEPackAndTransposeOptions packOptions;
    packOptions.packLevel = 0;
    funcPassManager.addPass(createAMDAIEPackAndTransposePass(packOptions));
  }

  // Promote the inputs and results to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 2;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::LinalgInputOutput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
    addCleanups();
  }

  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.tilingLevel = 2;
    tileFuseOptions.useSCFFor = true;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
    addCleanups();
  }

  LinalgFoldUnitExtentDimsPassOptions opts;
  opts.useRankReducingSlices = true;
  funcPassManager.addPass(mlir::createLinalgFoldUnitExtentDimsPass(opts));
  funcPassManager.addPass(createCanonicalizerPass());

  // Comprehensive bufferization
  addAMDAIEBufferizePasses(funcPassManager, useTilePipeline);
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
}

void addGeneralCopyPassPipeline(OpPassManager &funcPassManager,
                                TilePassPipeline useTilePipeline,
                                Operation *rootOp) {
  // Check if the root op is an elementwise operation.
  auto linalgRootOp = dyn_cast<linalg::LinalgOp>(rootOp);
  bool isElementwiseOp = linalgRootOp && isElementwise(linalgRootOp);

  auto addCleanups = [&]() {
    funcPassManager.addPass(createAMDAIECleanupPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  };

  // First level tiling using scf.forall.
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.hardwareMapping = HardwareMapping::Block;
    tileFuseOptions.tilingLevel = 0;
    tileFuseOptions.useSCFFor = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }

  // Insert copy operations.
  funcPassManager.addPass(createAMDAIEInsertCopyOpsPass());
  addCleanups();

  // Promote the input and result to shared memory.
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 1;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::LinalgInputOutput;
    bufferizeOptions.bufferizeElementwise = isElementwiseOp;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Second level tiling using scf.forall.
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.hardwareMapping = HardwareMapping::Core;
    tileFuseOptions.tilingLevel = 1;
    tileFuseOptions.useSCFFor = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }

  // Insert copy operations.
  funcPassManager.addPass(createAMDAIEInsertCopyOpsPass());
  addCleanups();

  // Promote the input and result to local memory.
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 2;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::LinalgInputOutput;
    bufferizeOptions.bufferizeElementwise = isElementwiseOp;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Tile the reduction dimension using scf.for.
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.tilingLevel = 2;
    tileFuseOptions.useSCFFor = true;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
    addCleanups();
  }

  // Lower to UKernels.
  funcPassManager.addPass(createAMDAIELowerToUKernelsPass());

  // Comprehensive bufferization
  addAMDAIEBufferizePasses(funcPassManager, useTilePipeline);
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
}

void buildAMDAIETransformPassPipeline(
    OpPassManager &variantPassManager, AMDAIEDevice device, uint32_t numRows,
    uint32_t numCols, TilePassPipeline useTilePipeline,
    LowerToAIEPassPipeline useLowerToAIEPipeline, bool matmulElementwiseFusion,
    bool enableVectorizationPasses, std::string enableAMDAIEUkernels,
    PacketFlowStrategy packetFlowStrategy, bool enableCoalescingLoops,
    bool enableCollapsingUnitDims, OutliningStrategy enableFunctionOutlining,
    int callReplication, bool insertLoopAroundCoreBlock, bool enableCtrlPkt,
    uint32_t coreStackSize) {
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPass(createTypePropagationPass)
        .addPass(createBubbleUpOrdinalOpsPass)
        .addPass(createBufferizeCopyOnlyDispatchesPass);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());
  {
    AMDAIELoweringStrategyOptions options;
    options.useTilePipeline = useTilePipeline;
    options.useLowerToAIEPipeline = useLowerToAIEPipeline;
    options.targetDevice = device;
    options.numRows = numRows;
    options.numCols = numCols;
    options.enableAMDAIEUkernels = enableAMDAIEUkernels;
    options.stackSize = coreStackSize;
    modulePassManager.addPass(createAMDAIELoweringStrategyPass(options));
  }
  modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    AMDAIELowerExecutableTargetOptions options;
    options.useTilePipeline = useTilePipeline;
    options.enableVectorizationPasses = enableVectorizationPasses;
    funcPassManager.addPass(
        [&]() { return createAMDAIELowerExecutableTargetPass(options); });
  }
  modulePassManager.addPass(createLowerUKernelOpsToCallsPass());
  if (useLowerToAIEPipeline == LowerToAIEPassPipeline::ObjectFifo) {
    addAMDAIEObjectFifoLoweringPasses(
        modulePassManager, packetFlowStrategy, useTilePipeline,
        enableVectorizationPasses, enableCoalescingLoops,
        enableCollapsingUnitDims, enableFunctionOutlining, callReplication,
        insertLoopAroundCoreBlock, numCols, enableCtrlPkt, coreStackSize);
  } else if (useLowerToAIEPipeline == LowerToAIEPassPipeline::AIR) {
    addMLIRAIRLoweringPasses(modulePassManager, device, useTilePipeline,
                             matmulElementwiseFusion,
                             enableVectorizationPasses);
  } else {
    assert(
        false &&
        "Only `ObjectFifo` and `AIR` pipelines supported for lowering to AIE");
  }
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  variantPassManager.addPass(createAMDAIELowerWorkgroupCountPass());

  LLVM_DEBUG({
    llvm::dbgs() << "Using AMDAIE pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

void addAMDAIEObjectFifoLoweringPasses(
    OpPassManager &passManager, PacketFlowStrategy packetFlowStrategy,
    TilePassPipeline useTilePipeline, bool enableVectorizationPasses,
    bool enableCoalescingLoops, bool enableCollapsingUnitDims,
    OutliningStrategy enableFunctionOutlining, int callReplication,
    bool insertLoopAroundCoreBlock, uint32_t numCols, bool enableCtrlPkt,
    uint32_t coreStackSize) {
  passManager.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());

  passManager.addPass(createAMDAIEDistributeL1AllocationsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(createCanonicalizerPass());
  // For matmul pipelines, we do transpose on target side for pack ops to get
  // better performance. While for convolution pipelines, the same settings
  // cause 'aie.dma_bd' error, so for now keep using transpose on source for
  // both pack and unpack ops.
  // TODO(vivian): explore the other options for conv ops.
  {
    AMDAIEConvertToDmaOptions dmaOptions;
    dmaOptions.packTransposeOnSource =
        (useTilePipeline == TilePassPipeline::ConvDecomposePipeline) ? true
                                                                     : false;
    dmaOptions.unpackTransposeOnSource = true;
    passManager.addPass(createAMDAIEConvertToDmaPass(dmaOptions));
  }

  passManager.addPass(createAMDAIENormalizeLoopBoundsPass());
  passManager.addPass(createAMDAIEInsertCoresPass({coreStackSize}));

  // Create function outlining options object, etc.
  {
    AMDAIELinalgFunctionOutliningOptions options;
    options.outliningStrategy = enableFunctionOutlining;
    passManager.addPass(createAMDAIELinalgFunctionOutliningPass(options));
  }
  {
    AMDAIEReplicateCallsOptions options;
    options.replication = callReplication;
    passManager.addPass(createAMDAIEReplicateCallsPass(options));
  }

  {
    // Vectorization passes
    OpPassManager &funcPassManager = passManager.nest<func::FuncOp>();
    appendVectorizationToPipeline(funcPassManager, enableVectorizationPasses,
                                  enableCoalescingLoops,
                                  enableCollapsingUnitDims);
  }

  passManager.addPass(createAMDAIELocalizeLogicalObjectFifoPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(createAMDAIEDistributeCoresAndObjectFifosPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIESplitLogicalObjFifosForConnectionReusePass());
  // Currently, SplitLogicalObjFifos pass has only been tested with the
  // following pipelines.
  if (useTilePipeline == TilePassPipeline::PackPeelPipeline ||
      useTilePipeline == TilePassPipeline::PackPeel4LevelTilingPipeline ||
      useTilePipeline == TilePassPipeline::GeneralCopyPipeline)
    passManager.addPass(createAMDAIESplitLogicalObjFifosPass());

  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createAMDAIEAssignLogicalObjectFifoDepthPass());

  passManager.addPass(createAMDAIEAssignTilesPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIEDmaToCircularDmaPass());
  passManager.addNestedPass<func::FuncOp>(createAMDAIECreateAIEWorkgroupPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createAMDAIEDmaCSEPass());

  passManager.addPass(createAMDAIEHoistLogicalObjFifoPass());
  passManager.addPass(createAMDAIECanonicalizeDoublyStridedOpPass());
  passManager.addPass(createAMDAIEFlattenLogicalObjectFifoPass());
  passManager.addPass(createAMDAIEAccessToAcquireReleasePass());
  passManager.addPass(createAMDAIENoneAccessToTemporaryBufferPass());

  {
    AMDAIEGenerateControlOverlayOptions options;
    options.routeShimToTileCtrl = enableCtrlPkt;
    passManager.addPass(createAMDAIEGenerateControlOverlayPass(options));
    passManager.addPass(createCSEPass());
    passManager.addPass(createCanonicalizerPass());
  }

  {
    AMDAIEAssignConnectionTypesOptions options;
    options.packetFlowStrategy = packetFlowStrategy;
    passManager.addPass(createAMDAIEAssignConnectionTypesPass(options));
    passManager.addPass(createCSEPass());
    passManager.addPass(createCanonicalizerPass());
  }

  // Convert control code `scf.forall` ops to `scf.for` ops right before the DMA
  // composition optimization pass to enable more loop subsumption optimization
  // opportunities.
  passManager.addPass(createAMDAIEControlCodeForallToForPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createAMDAIEDmaCompositionPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createAMDAIEDmaCSEPass());

  passManager.addPass(createAMDAIEAssignChannelsPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIEAssignNpuDmaBdIdsPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIEControlCodeLoopUnrollPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createAMDAIEDmaCSEPass());

  passManager.addPass(createAMDAIECanonicalizeDoublyStridedOpPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIEConvertCoreForallToForPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIEObjFifoBufferizationPass());
  passManager.addPass(createAMDAIETemporaryAllocBufferizationPass());

  passManager.addPass(createAMDAIEConnectionToFlowPass());
  passManager.addPass(createAMDAIEAssignPacketIdsPass());

  passManager.addPass(createAMDAIENpuDmaToHalfDmaCpyNdPass());
  passManager.addPass(createAMDAIEInsertDmaBdChainPass());
  passManager.addPass(createAMDAIEFoldDmaWaitsPass());
  passManager.addPass(createAMDAIEControlCodeLoweringPass());
  passManager.addPass(createAMDAIEControlCodeToTransactionPass());

  addAMDAIEToAIEPasses(passManager, insertLoopAroundCoreBlock);

  // Now lower using the AIE passes from MLIR-AIE.
  addMLIRAIELoweringPasses(passManager, useTilePipeline);
}

void addMLIRAIELoweringPasses(OpPassManager &pm,
                              TilePassPipeline useTilePipeline) {
  mlir::iree_compiler::aievec::buildConvertVectorToAIEVec(pm);

  {
    OpPassManager &devicePM = pm.nest<xilinx::AIE::DeviceOp>();
    devicePM.addPass(createCanonicalizerPass());
    devicePM.addPass(createAMDAIEAssignBufferDescriptorIDsPass());
    {
      // For Conv ops use basic sequential scheme to avoid numerical error.
      // TODO: Find a better working scheme for Conv ops
      AMDAIEAssignBufferAddressesOptions options;
      if (useTilePipeline == TilePassPipeline::ConvDecomposePipeline)
        options.allocScheme = AllocScheme::Sequential;
      devicePM.addPass(createAMDAIEAssignBufferAddressesPass(options));
    }
    {
      // Route control and data flows separately, prioritizing control flows
      // first to ensure their deterministic routing results.
      AMDAIERouteFlowsWithPathfinderOptions options;
      // Route only control flows.
      options.routeCtrl = true;
      options.routeData = false;
      devicePM.addPass(createAMDAIERouteFlowsWithPathfinderPass(options));
      // Route only data flows.
      options.routeCtrl = false;
      options.routeData = true;
      devicePM.addPass(createAMDAIERouteFlowsWithPathfinderPass(options));
    }
  }

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createSCFToControlFlowPass());

  {
    OpPassManager &devicePM = pm.nest<xilinx::AIE::DeviceOp>();
    devicePM.addPass(createAMDAIELocalizeLocksPass());
    devicePM.addPass(createAMDAIENormalizeAddressSpacesPass());
    devicePM.addPass(createCanonicalizerPass());
  }

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(aievec::createConvertAIEVecToLLVMPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createAMDAIELoadStoreAlignmentResetPass());
  pm.addPass(createCanonicalizerPass());
}

// TODO (Erwei): The "packPeel" temporary argument should be removed once
// pack-peel and pack-pad share the same pass pipeline. See TODOs inlined below
// for details.
void addMLIRAIRLoweringPasses(OpPassManager &passManager, AMDAIEDevice device,
                              TilePassPipeline useTilePipeline,
                              bool matmulElementwiseFusion,
                              bool enableVectorizationPasses) {
  // Add passes for preparing for lowering to MLIR-AIR
  passManager.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());
  {
    // Vectorization passes
    OpPassManager &funcPassManager = passManager.nest<func::FuncOp>();
    appendVectorizationToPipeline(funcPassManager, enableVectorizationPasses);
  }
  passManager.addPass(createAMDAIEBridgeToAIRPass());

  // Running canonicalization for all pipelines here results in failures.
  // Example
  // ```
  // 'memref.cast' op is an unsupported operation. This pass currently only
  // supports AllocOp and SubViewOp as inputs.
  // ```
  // It is currently required for the convolution pipeline though, to remove the
  // extra (size-1) thread- and group- dimensions.
  //
  // TODO(newling) there are better solutions like:
  // 1) make canonicalization work for scf.forall
  // 2) pass to collapse rank-4 scf.foralls to rank-2 scf.foralls.
  // 3) resolve above 'unsupproted operation' error.
  if (useTilePipeline == TilePassPipeline::ConvDecomposePipeline) {
    passManager.addPass(createCanonicalizerPass());
  }

  passManager.addPass(createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass());

  // TODO(newling) adding createCanonicalizerPass introduces a dma copy lowering
  // failure. Understand and fix.
  passManager.addPass(createCSEPass());
  {
    xilinx::air::ParallelToHerdOptions options;
    options.clAssignDepth = -1;
    passManager.addPass(xilinx::air::createParallelToHerdPass(options));
  }
  {
    xilinx::air::ParallelToLaunchOptions options;
    options.clHasSegment = true;
    options.clAssignDepth = 0;
    passManager.addPass(xilinx::air::createParallelToLaunchPass(options));
  }
  passManager.addPass(mlir::createForallToForLoopPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(xilinx::air::createCopyToDmaPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(xilinx::air::createAIRDependencyPass());
  if (!(useTilePipeline == TilePassPipeline::PackPeelPipeline &&
        matmulElementwiseFusion)) {
    passManager.addPass(xilinx::air::createAIRBroadcastDetection());
    passManager.addPass(xilinx::air::createAIRHoistDmaInAccumPattern());
    passManager.addPass(xilinx::air::createAIRSpecializeDmaBroadcast());
  }
  passManager.addPass(xilinx::air::createDmaToChannelPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(xilinx::air::createAIRDependencyCanonicalizePass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(xilinx::air::createAIRIsolateAsyncDmaLoopNests());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  {
    xilinx::air::AIRFuseChannelsOptions options;
    if (useTilePipeline == TilePassPipeline::PackPeelPipeline &&
        matmulElementwiseFusion) {
      const static llvm::SmallVector<std::string> mode = {"L1"};
      options.clAggressiveMode = mode;
    }
    passManager.addPass(xilinx::air::createAIRFuseChannels(options));
  }
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(
      xilinx::air::createAIRSplitL2MemrefForBufferConstraintPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(xilinx::air::createAIRIsolateAsyncDmaLoopNests());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(
      xilinx::air::createAIRFuseAllocDealloc());
  passManager.addNestedPass<func::FuncOp>(
      xilinx::air::createAIRShrinkMemrefSizesByAccess());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  if (useTilePipeline != TilePassPipeline::PackPeel4LevelTilingPipeline) {
    passManager.addPass(
        xilinx::air::createAIRLabelScfForLoopForPingPongPattern());
    {
      xilinx::air::AIRPingPongTransformationPatternOptions options;
      options.clKeepMemrefDealloc = true;
      passManager.addPass(
          xilinx::air::createAIRPingPongTransformationPattern(options));
    }
    passManager.addPass(createCanonicalizerPass());
    passManager.addPass(createCSEPass());
  }
  {
    xilinx::air::AIROptimizeMemtileDMABDsOptions options;
    options.clDevice = stringifyEnum(device);
    passManager.addNestedPass<func::FuncOp>(
        xilinx::air::createAIROptimizeMemtileDMABDs(options));
  }

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  {
    xilinx::air::AIRCollapseHerdPassOptions options;
    options.clMaxColSize = 4;
    passManager.addNestedPass<func::FuncOp>(
        xilinx::air::createAIRCollapseHerdPass(options));
  }
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  {
    xilinx::air::AIRHerdPlacementPassOptions options;
    options.clNumRows = 4;
    options.clNumCols = 4;
    options.clAnchorPointRow = 2;
    options.clAnchorPointCol = 0;
    passManager.addPass(xilinx::air::createAIRHerdPlacementPass(options));
  }
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addNestedPass<func::FuncOp>(
      xilinx::air::createAIRRenumberDmaIdPass());
  passManager.addNestedPass<func::FuncOp>(
      mlir::createConvertLinalgToLoopsPass());

  {
    xilinx::air::AIRToAIEOptions options;
    options.clRowOffset = 2;
    options.clColOffset = 0;
    options.clDevice = stringifyEnum(device);
    options.clEmitWhileLoop = true;
    passManager.addPass(xilinx::air::createAIRToAIEPass(options));
  }
  {
    xilinx::air::AIROptimizeShimDMABDsOptions options;
    options.clDevice = stringifyEnum(device);
    const static llvm::SmallVector<unsigned> tile_sizes = {2, 2};
    options.clTileSizes = tile_sizes;
    passManager.addNestedPass<func::FuncOp>(
        xilinx::air::createAIROptimizeShimDMABDs(options));
  }
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(xilinx::air::createAIRLoweringPass());
  passManager.addPass(mlir::affine::createAffineExpandIndexOpsPass());
  passManager.addPass(createAMDAIELowerFuncArgsPass());
  passManager.addPass(xilinx::airrt::createAIRRtToNpuPass());

  {
    // `AMDAIEDmaToNpuPass` and `AMDAIEIncrementRepeatCountPass` are only needed
    // for AIR.
    OpPassManager &devicePM = passManager.nest<xilinx::AIE::DeviceOp>();
    devicePM.addPass(createCanonicalizerPass());
    devicePM.addPass(createAMDAIEDmaToNpuPass());
    devicePM.addPass(createAMDAIEIncrementRepeatCountPass());
  }

  // Now lower using the AIE passes from MLIR-AIE.
  addMLIRAIELoweringPasses(passManager, useTilePipeline);
}

// NOTE: this runs on the top-level program module containing all hal.executable
// ops.
void buildAMDAIELinkingPassPipeline(OpPassManager &passManager) {
  // Link together executables. This may produce some IR duplication.
  passManager.addPass(createAMDAIELinkExecutablesPass());

  // Cleanup IR duplication.
  passManager.addNestedPass<IREE::HAL::ExecutableOp>(
      mlir::createCanonicalizerPass());
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree-amd-aie/Transforms/Passes.h.inc"
}  // namespace

void registerAMDAIEPasses() {
  // Generated.
  registerPasses();
}

}  // namespace mlir::iree_compiler::AMDAIE
