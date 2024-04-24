// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"

#include "air/Conversion/Passes.h"
#include "air/Transform/Passes.h"
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

void appendVectorizationToPipeline(OpPassManager &funcPassManager) {
  if (!clEnableVectorizationPasses) return;
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createAMDAIEInsertLoopsForVectorizationPass());
  funcPassManager.addPass(createAMDAIEVectorizationPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

//===---------------------------------------------------------------------===//
// Default allocation functions for AIE backend
//===---------------------------------------------------------------------===//
// Allocation callbacks to use with upstream comprehensive bufferization
static FailureOr<Value> aieComprehensiveBufferizeAllocationFn(
    OpBuilder &builder, Location loc, MemRefType memRefType,
    ValueRange dynamicSizes, unsigned alignment) {
  return builder
      .create<memref::AllocaOp>(loc, memRefType, dynamicSizes,
                                builder.getI64IntegerAttr(alignment))
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
    AMDAIETileAndFuseOptions tileFuseOptions0;
    tileFuseOptions0.tilingLevel = 0;
    tileFuseOptions0.useSCFFor = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions0));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // First level packing
  {
    AMDAIEPackAndTransposeOptions packOptions0;
    packOptions0.packLevel = 0;
    funcPassManager.addPass(createAMDAIEPackAndTransposePass(packOptions0));
  }

  // Propagate pack ops for the elementwise op
  funcPassManager.addPass(createAMDAIEPropagateDataLayoutPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the output to shared memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions0;
    bufferizeOptions0.memorySpace = 1;
    bufferizeOptions0.bufferizeOperand = BufferizeOperand::Output;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions0));
  }

  // Second level packing
  {
    AMDAIEPackAndTransposeOptions packOptions1;
    packOptions1.packLevel = 1;
    funcPassManager.addPass(createAMDAIEPackAndTransposePass(packOptions1));
  }

  // Propagate pack ops for the elementwise op
  funcPassManager.addPass(createAMDAIEPropagateDataLayoutPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the output to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions1;
    bufferizeOptions1.memorySpace = 2;
    bufferizeOptions1.bufferizeOperand = BufferizeOperand::Output;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions1));
  }

  // Promote the operands from elementwise op to shared and local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions2;
    bufferizeOptions2.memorySpace = 1;
    bufferizeOptions2.bufferizeElementwise = true;
    bufferizeOptions2.bufferizeOperand = BufferizeOperand::DefOp;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions2));
  }
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions3;
    bufferizeOptions3.memorySpace = 2;
    bufferizeOptions3.bufferizeElementwise = true;
    bufferizeOptions3.bufferizeOperand = BufferizeOperand::InputOutput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions3));
  }

  // Tile the reduction dimension using scf.for
  {
    AMDAIETileAndFuseOptions tileFuseOptions1;
    tileFuseOptions1.tilingLevel = 1;
    tileFuseOptions1.useSCFFor = true;
    tileFuseOptions1.tileElementwise = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions1));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Fuse both levels of pack ops into for loop
  {
    AMDAIEFusePackIntoLoopOptions fusePackOptions0;
    fusePackOptions0.fusePackDepth = 2;
    fusePackOptions0.useSCFFor = true;
    funcPassManager.addPass(createAMDAIEFusePackIntoLoopPass(fusePackOptions0));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the inputs to shared memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions4;
    bufferizeOptions4.memorySpace = 1;
    bufferizeOptions4.bufferizeOperand = BufferizeOperand::DefOp;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions4));
  }

  // Second level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions2;
    tileFuseOptions2.tilingLevel = 2;
    tileFuseOptions2.useSCFFor = false;
    tileFuseOptions2.tileElementwise = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions2));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Fuse second level pack ops into forall loop
  {
    AMDAIEFusePackIntoLoopOptions fusePackOptions1;
    fusePackOptions1.fusePackDepth = 1;
    fusePackOptions1.useSCFFor = false;
    funcPassManager.addPass(createAMDAIEFusePackIntoLoopPass(fusePackOptions1));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Promote the inputs to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions5;
    bufferizeOptions5.memorySpace = 2;
    bufferizeOptions5.bufferizeOperand = BufferizeOperand::Input;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions5));
  }

  // Hoist static allocations
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Peel the first iteration out of the for loop.
  // TODO (vivian): Find a way to automatically detect matmul + elementwise
  // dispatches, so that we can change the peelOptions to peel both first and
  // last iterations.
  {
    AMDAIEPeelForLoopOptions peelOptions;
    peelOptions.peelingType = PeelingType::First;
    funcPassManager.addPass(createAMDAIEPeelForLoopPass(peelOptions));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Fuse fill into forall loop
  funcPassManager.addPass(createAMDAIEFuseFillIntoForallPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Comprehensive bufferization
  addAMDAIEBufferizePasses(funcPassManager);
}

void addPadPackBasedPassPipeline(OpPassManager &funcPassManager,
                                 TilingConfig &tilingConfig) {
  // First level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions0;
    tileFuseOptions0.tilingLevel = 0;
    tileFuseOptions0.useSCFFor = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions0));
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
    AMDAIEBufferizeToAllocationOptions bufferizeOptions0;
    bufferizeOptions0.memorySpace = 1;
    bufferizeOptions0.bufferizeOperand = BufferizeOperand::InputOutput;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions0));
  }

  // Tile linalg.copy ops using scf.for
  {
    AMDAIETileOptions tileOptions1;
    tileOptions1.tilingLevel = 1;
    funcPassManager.addPass(createAMDAIETilePass(tileOptions1));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Second level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions2;
    tileFuseOptions2.tilingLevel = 2;
    tileFuseOptions2.useSCFFor = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions2));
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
    AMDAIEBufferizeToAllocationOptions bufferizeOptions1;
    bufferizeOptions1.memorySpace = 2;
    bufferizeOptions1.bufferizeOperand = BufferizeOperand::Output;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions1));
  }

  // Tile the reduction dimension using scf.for
  {
    AMDAIETileAndFuseOptions tileFuseOptions3;
    tileFuseOptions3.tilingLevel = 3;
    tileFuseOptions3.useSCFFor = true;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions3));
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
    AMDAIEBufferizeToAllocationOptions bufferizeOptions2;
    bufferizeOptions2.memorySpace = 2;
    bufferizeOptions2.bufferizeOperand = BufferizeOperand::Input;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions2));
  }

  // Lower to UKernels
  {
    AMDAIELowerToUKernelsOptions options;
    options.passPipeline = AIEPassPipeline::PadPackPipeline;
    options.pathToUkernels = clPathToUkernels;
    funcPassManager.addPass(createAMDAIELowerToUKernelsPass(options));
  }
  // Vectorization passes
  appendVectorizationToPipeline(funcPassManager);

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
    addMLIRAIRAIELoweringPasses(modulePassManager);
  } else if (clUsePipeline != AIEPassPipeline::PackPeelPipeline) {
    addMLIRAIRAIELegacyLoweringPasses(modulePassManager);
  }
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  variantPassManager.addPass(createAMDAIELowerWorkgroupCountPass());

  LLVM_DEBUG({
    llvm::dbgs() << "Using AMDAIE pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

void addMLIRAIRAIELoweringPasses(OpPassManager &passManager) {
  // Add passes for preparing for lowering to MLIR-AIR
  passManager.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());
  passManager.addPass(createAMDAIEBridgeToAIRPass());
  passManager.addPass(createAMDAIEPackToDmaPass());

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
  passManager.addPass(xilinx::air::createAIRDependencyScheduleOptPass());
  passManager.addPass(xilinx::air::createAIRSpecializeDmaBroadcast());
  passManager.addPass(xilinx::air::createDmaToChannelPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(xilinx::air::createAIRDependencyCanonicalizePass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(
      xilinx::air::createAIRSplitL2MemrefForBufferConstraintPass());
  passManager.addPass(xilinx::air::createAIRIsolateAsyncDmaLoopNests());
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
    options.clDevice = "ipu";
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

  passManager.addPass(xilinx::airrt::createAIRRtToIpuPass());
  passManager.addPass(createCanonicalizerPass());
}

void addMLIRAIRAIELegacyLoweringPasses(OpPassManager &passManager) {
  // Add passes for preparing for lowering to MLIR-AIR
  passManager.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());
  passManager.addPass(createAMDAIEBridgeToAIRPass());
  passManager.addPass(createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass());

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
  passManager.addPass(xilinx::air::createCopyToDmaPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(xilinx::air::createAIRDependencyPass());
  passManager.addPass(xilinx::air::createAIRDependencyScheduleOptPass());
  passManager.addPass(xilinx::air::createAIRSpecializeDmaBroadcast());
  passManager.addPass(xilinx::air::createDmaToChannelPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(xilinx::air::createAIRDependencyCanonicalizePass());
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

  passManager.addPass(xilinx::air::createAIRIsolateAsyncDmaLoopNests());
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
    options.clNumCols = 1;
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
    options.clDevice = "ipu";
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

  passManager.addPass(xilinx::airrt::createAIRRtToIpuPass());
  passManager.addPass(createCanonicalizerPass());
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree-amd-aie/Transforms/Passes.h.inc"
}  // namespace

void registerAMDAIEPasses() {
  // Generated.
  registerPasses();

  static PassPipelineRegistration<> AIELoweringPipeline(
      "iree-amdaie-aie-lowering-pipeline",
      "Runs the AIR/AIE lowering passes to tiled and distributed code",
      [](OpPassManager &passManager) {
        if (clUsePipeline == AIEPassPipeline::PadPackPipeline) {
          addMLIRAIRAIELoweringPasses(passManager);
        } else {
          addMLIRAIRAIELegacyLoweringPasses(passManager);
        }
      });
}

}  // namespace mlir::iree_compiler::AMDAIE
