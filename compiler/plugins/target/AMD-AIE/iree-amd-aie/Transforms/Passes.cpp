// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"

#include "air/Conversion/Passes.h"
#include "air/Transform/Passes.h"
#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-amdaie-lowering-pass-pipeline"

namespace mlir::iree_compiler::AMDAIE {

/// Command line options used purely for development purposes. Not to be relied
/// on in any way.
static llvm::cl::opt<AIEPassPipeline> clUsePipeline(
    "iree-amdaie-use-pipeline",
    llvm::cl::desc("Pick the lowering pipeline to use"),
    llvm::cl::values(
        clEnumValN(
            AIEPassPipeline::PackPeelPipeline, "pack-peel",
            "Use the IREE lowering to AIR dialect through pack operation"),
        clEnumValN(AIEPassPipeline::PadPackPipeline, "pad-pack",
                   "Use the IREE lowering to AIR dialect through "
                   "pad and pack operations")),
    llvm::cl::init(AIEPassPipeline::PadPackPipeline));

static llvm::cl::opt<int32_t> clNumCores(
    "iree-amdaie-num-cores",
    llvm::cl::desc("Choose the number of cores to use"), llvm::cl::init(1));

static llvm::cl::opt<std::string> clPathToUkernels(
    "iree-amdaie-path-to-ukernels",
    llvm::cl::desc("Path to microkernels' directory"));

static llvm::cl::opt<bool> clEnableVectorizationPasses(
    "iree-amdaie-enable-vectorization-passes",
    llvm::cl::desc("Some pipelines (see iree-amdaie-use-pipeline) may include "
                   "vectorization passes. This option enables or disables "
                   "these vectorization passes. It is intended for development "
                   "purposes only."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clMatmulElementwiseFusion(
    "iree-amdaie-matmul-elementwise-fusion",
    llvm::cl::desc("This option enables/disables special passes in MLIR-AIR "
                   "for matmul-elementwise fusion. It is currently added for "
                   "development purpose and should be removed in the future."),
    llvm::cl::init(false));

void appendVectorizationToPipeline(OpPassManager &funcPassManager) {
  if (!clEnableVectorizationPasses) return;
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createAMDAIEInsertLoopsForVectorizationPass());
  funcPassManager.addPass(createAMDAIEVectorizationPass());
}

//===---------------------------------------------------------------------===//
// Default allocation functions for AIE backend
//===---------------------------------------------------------------------===//
// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> aieComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  int64_t numDims = memRefType.getShape().size();
  AMDAIEMemSpace memSpace = AMDAIEMemSpace::Local;
  if (clUsePipeline == AIEPassPipeline::PackPeelPipeline && numDims == 4) {
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
}

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

static void addAMDAIEBufferizePasses(OpPassManager &pm) {
  // Bufferize the dispatch.
  using mlir::bufferization::BufferizationOptions;
  BufferizationOptions::AllocationFn allocationFn =
      aieComprehensiveBufferizeAllocationFn;
  BufferizationOptions::MemCpyFn memCpyFn = aieComprehensiveBufferizeCopyFn;
  addIREEComprehensiveBufferizePasses(pm, allocationFn, memCpyFn);
}

void addPackPeelBasedPassPipeline(OpPassManager &funcPassManager,
                                  TilingConfig &tilingConfig) {
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
    bufferizeOptions.bufferizeOperand = BufferizeOperand::Output;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Promote the elementwise input to shared memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 1;
    bufferizeOptions.bufferizeElementwise = true;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::Input;
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

  // Promote the matmul output to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 2;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::Output;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

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
    AMDAIEFusePackIntoLoopOptions fusePackOptions;
    fusePackOptions.fusePackDepth = 2;
    fusePackOptions.useSCFFor = true;
    funcPassManager.addPass(createAMDAIEFusePackIntoLoopPass(fusePackOptions));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the matmul inputs to shared memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 1;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::DefOp;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Second level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
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
    AMDAIEFusePackIntoLoopOptions fusePackOptions;
    fusePackOptions.fusePackDepth = 1;
    fusePackOptions.useSCFFor = false;
    funcPassManager.addPass(createAMDAIEFusePackIntoLoopPass(fusePackOptions));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the matmul inputs to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 2;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::Input;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

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
    AMDAIEFusePackIntoLoopOptions fusePackOptions;
    fusePackOptions.fusePackDepth = 1;
    fusePackOptions.useSCFFor = false;
    fusePackOptions.targetElementwise = true;
    funcPassManager.addPass(createAMDAIEFusePackIntoLoopPass(fusePackOptions));
  }

  // Promote the elementwise input to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 2;
    bufferizeOptions.bufferizeElementwise = true;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::Input;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Lower to UKernels.
  {
    AMDAIELowerToUKernelsOptions options;
    options.pathToUkernels = clPathToUkernels;
    funcPassManager.addPass(createAMDAIELowerToUKernelsPass(options));
  }

  // Vectorization passes
  appendVectorizationToPipeline(funcPassManager);

  // Comprehensive bufferization
  addAMDAIEBufferizePasses(funcPassManager);
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
}

void addPadPackBasedPassPipeline(OpPassManager &funcPassManager,
                                 TilingConfig &tilingConfig) {
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

  // Pad the linalg operation
  {
    AMDAIEPadOptions padOptions;
    padOptions.paddingLevel = 0;
    funcPassManager.addPass(createAMDAIEPadPass(padOptions));
  }

  // Promote the input and result to shared memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 1;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::InputOutput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Tile linalg.copy ops using scf.for
  {
    AMDAIETileOptions tileOptions;
    tileOptions.tilingLevel = 1;
    funcPassManager.addPass(createAMDAIETilePass(tileOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Second level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.tilingLevel = 2;
    tileFuseOptions.useSCFFor = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Pack the linalg operation
  {
    AMDAIEPackAndTransposeOptions packOptions;
    packOptions.packLevel = 0;
    funcPassManager.addPass(createAMDAIEPackAndTransposePass(packOptions));
  }

  // Only promote the result to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 2;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::Output;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Tile the reduction dimension using scf.for
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.tilingLevel = 3;
    tileFuseOptions.useSCFFor = true;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Fuse pack ops into for loop
  funcPassManager.addPass(createAMDAIEFusePackIntoLoopPass());
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the inputs to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 2;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::Input;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Lower to UKernels
  {
    AMDAIELowerToUKernelsOptions options;
    options.pathToUkernels = clPathToUkernels;
    funcPassManager.addPass(createAMDAIELowerToUKernelsPass(options));
  }
  // Vectorization passes
  appendVectorizationToPipeline(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());

  // Comprehensive bufferization
  addAMDAIEBufferizePasses(funcPassManager);
}

void buildAMDAIETransformPassPipeline(OpPassManager &variantPassManager) {
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());
  {
    AMDAIELoweringStrategyOptions options;
    options.usePassPipeline = clUsePipeline;
    options.numCores = clNumCores;
    modulePassManager.addPass(createAMDAIELoweringStrategyPass(options));
  }
  modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    AMDAIELowerExecutableTargetOptions options;
    options.usePassPipeline = clUsePipeline;
    funcPassManager.addPass(
        [&]() { return createAMDAIELowerExecutableTargetPass(options); });
  }
  modulePassManager.addPass(createLowerUKernelOpsToCallsPass());
  if (clUsePipeline == AIEPassPipeline::PadPackPipeline) {
    addMLIRAIRAIELoweringPasses(modulePassManager, false);
  } else if (clUsePipeline == AIEPassPipeline::PackPeelPipeline) {
    addMLIRAIRAIELoweringPasses(modulePassManager, true);
  }
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  variantPassManager.addPass(createAMDAIELowerWorkgroupCountPass());

  LLVM_DEBUG({
    llvm::dbgs() << "Using AMDAIE pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

/// TODO(avarma): Currently adding this as a separate utility until we deprecate
/// the rest and keep just this as the main set of lowering passes.
static void addAIELoweringPassesForPackPeel(OpPassManager &passManager) {
  passManager.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());
  passManager.addPass(createAMDAIEPackToDmaPass());
  passManager.addPass(xilinx::air::createCopyToDmaPass());

  passManager.addPass(createAMDAIEAIRDmaAMDAIEDmaPass());
  passManager.addPass(createAMDAIENormalizeLoopBoundsPass());
  passManager.addPass(createAMDAIEInsertCoresPass());
  passManager.addPass(createAMDAIELocalizeLogicalObjectFifoPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(createAMDAIEDistributeCoresAndObjectFifosPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIEDmaToCircularDmaPass());
  passManager.addNestedPass<func::FuncOp>(createAMDAIECreateAIEWorkgroupPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(createAMDAIECanonicalizeDoublyStridedOpPass());
  passManager.addPass(createAMDAIEAccessToAcquireReleasePass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIEControlCodeLoopUnrollPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIECreateLogicalObjectFifoLinkPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIELowerToAIEPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createConvertLinalgToLoopsPass());
}

// TODO (Erwei): The "packPeel" temporary argument should be removed once
// pack-peel and pack-pad share the same pass pipeline. See TODOs inlined below
// for details.
void addMLIRAIRAIELoweringPasses(OpPassManager &passManager, bool packPeel) {
  if (packPeel) {
    addAIELoweringPassesForPackPeel(passManager);
    return;
  }
  // Add passes for preparing for lowering to MLIR-AIR
  passManager.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());
  passManager.addPass(createAMDAIEBridgeToAIRPass());
  passManager.addPass(createAMDAIEPackToDmaPass());
  // TODO (Erwei): Figure out a way to work with AMDAIEPackToDmaPass.
  if (packPeel)
    passManager.addPass(createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass());
  else
    passManager.addPass(createAMDAIEPackToDmaPass());

  // TODO(newling) adding createCanonicalizerPass introduces a dma copy lowering
  // failure. Understand and fix.
  passManager.addPass(createCSEPass());
  {
    xilinx::air::ParallelToHerdOptions options;
    options.clAssignDepth = 1;
    passManager.addPass(xilinx::air::createParallelToHerdPass(options));
  }
  {
    xilinx::air::ParallelToLaunchOptions options;
    options.clHasSegment = true;
    passManager.addPass(xilinx::air::createParallelToLaunchPass(options));
  }
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createAMDAIECanonicalizeDmaPass());
  passManager.addPass(xilinx::air::createCopyToDmaPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(xilinx::air::createAIRDependencyPass());
  if (!(packPeel && clMatmulElementwiseFusion)) {
    passManager.addPass(xilinx::air::createAIRDependencyScheduleOptPass());
    passManager.addPass(xilinx::air::createAIRSpecializeDmaBroadcast());
  }
  passManager.addPass(xilinx::air::createDmaToChannelPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(xilinx::air::createAIRDependencyCanonicalizePass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  // TODO (Erwei): This pass currently doesn't support pack-peel pipeline. This
  // pass needs to work in order to get multiple AIE columns to work.
  if (!packPeel)
    passManager.addNestedPass<func::FuncOp>(
        xilinx::air::createAIRSplitL2MemrefForBufferConstraintPass());
  passManager.addPass(xilinx::air::createAIRIsolateAsyncDmaLoopNests());
  // TODO (Erwei): Check for this pass's stability, to ensure backward
  // compatibility with pad-pack pipeline.
  if (packPeel) {
    passManager.addPass(createCanonicalizerPass());
    passManager.addPass(createCSEPass());
    {
      xilinx::air::AIRFuseChannelsOptions options;
      std::vector<std::string> mode;
      if (packPeel && clMatmulElementwiseFusion) {
        mode.push_back("L1");
      }
      options.clAggressiveMode = ArrayRef(mode);
      passManager.addPass(xilinx::air::createAIRFuseChannels(options));
    }
  }
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(
      xilinx::air::createAIRSegmentLoopFusion());

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

  passManager.addPass(
      xilinx::air::createAIRSpecializeChannelWrapAndStridePattern());
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
    options.clDevice = "npu1_4col";
    options.clEmitWhileLoop = true;
    passManager.addPass(xilinx::air::createAIRToAIEPass(options));
  }
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(xilinx::air::createAIRLoweringPass());
  {
    xilinx::air::AffineLoopOptPassOptions options;
    const std::vector<unsigned> tile_sizes = {4, 4};
    options.clTileSizes = ArrayRef(tile_sizes);
    passManager.addNestedPass<func::FuncOp>(
        xilinx::air::createAffineLoopOptPass(options));
  }
  {
    xilinx::air::AIRUnrollOuterPerfectlyNestedLoopsPassOptions options;
    options.clDepth = 2;
    passManager.addNestedPass<func::FuncOp>(
        xilinx::air::createAIRUnrollOuterPerfectlyNestedLoopsPass(options));
  }
  passManager.addPass(mlir::affine::createAffineExpandIndexOpsPass());

  passManager.addPass(xilinx::airrt::createAIRRtToNpuPass());
  passManager.addPass(createCanonicalizerPass());
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
