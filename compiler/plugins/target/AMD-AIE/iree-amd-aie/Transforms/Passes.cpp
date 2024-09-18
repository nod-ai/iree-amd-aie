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
#include "air/Transform/AffineLoopOptPass.h"
#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-amdaie-lowering-pass-pipeline"

namespace mlir::iree_compiler::AMDAIE {

/// Command line option for selecting the lowering pipeline to use to generate
/// AIE DMA configurations, core code and control code.
static llvm::cl::opt<LowerToAIEPassPipeline> clUseLowerToAIEPipeline(
    "iree-amdaie-lower-to-aie-pipeline",
    llvm::cl::desc("Pick the lowering pipeline to use"),
    llvm::cl::values(clEnumValN(LowerToAIEPassPipeline::AIR, "air",
                                "Use the IREE lowering through AIR"),
                     clEnumValN(LowerToAIEPassPipeline::ObjectFifo,
                                "objectFifo",
                                "Use the IREE lowering to objectFifos")),
    llvm::cl::init(LowerToAIEPassPipeline::ObjectFifo));

/// Command line option for selecting the lowering pipeline to use tiling
/// computations and packing data.
static llvm::cl::opt<TilePassPipeline> clUseTilePipeline(
    "iree-amdaie-tile-pipeline",
    llvm::cl::desc("Pick the lowering pipeline to use"),
    llvm::cl::values(
        clEnumValN(
            TilePassPipeline::PackPeelPipeline, "pack-peel",
            "Use the pack-peel based lowering strategy for matmul-like ops"),
        clEnumValN(
            TilePassPipeline::PadPackPipeline, "pad-pack",
            "Use the pad-pack based lowering strategy for matmul-like ops"),
        clEnumValN(TilePassPipeline::ConvDecomposePipeline, "conv-decompose",
                   "Use the conv-decompose based lowering strategy for "
                   "convolution interface ops")),
    llvm::cl::init(TilePassPipeline::PackPeelPipeline));

static llvm::cl::opt<std::string> clPathToUkernels(
    "iree-amdaie-path-to-ukernels",
    llvm::cl::desc("Path to microkernels' directory"), llvm::cl::init(""));

static llvm::cl::opt<bool> clEnableVectorizationPasses(
    "iree-amdaie-enable-vectorization-passes",
    llvm::cl::desc("Some pipelines (see iree-amdaie-tile-pipeline) may include "
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
  if (clUseTilePipeline == TilePassPipeline::PackPeelPipeline && numDims == 4) {
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

void addAMDAIEToAIEPasses(OpPassManager &passManager) {
  passManager.addPass(createAMDAIEAcquireReleaseToUseLockPass());
  passManager.addPass(createAMDAIECanonicalizeNpuDmaCpyNdPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createAMDAIESinkIntoCorePass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createAMDAIELowerToAIEPass());
  passManager.addPass(createAMDAIERemoveMemorySpacePass());
  passManager.addPass(createCanonicalizerPass());
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
    // windows
    options.pathToUkernels = escapeCommandLineComponent(clPathToUkernels);
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
    // windows
    options.pathToUkernels = escapeCommandLineComponent(clPathToUkernels);
    funcPassManager.addPass(createAMDAIELowerToUKernelsPass(options));
  }
  // Vectorization passes
  appendVectorizationToPipeline(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());

  // Comprehensive bufferization
  addAMDAIEBufferizePasses(funcPassManager);
}

void addConvDecomposePassPipeline(OpPassManager &funcPassManager,
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

  // Second level tiling using scf.forall
  {
    AMDAIETileAndFuseOptions tileFuseOptions;
    tileFuseOptions.tilingLevel = 1;
    tileFuseOptions.useSCFFor = false;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Fuse fill op into the inner forall loop
  funcPassManager.addPass(createAMDAIEFuseFillIntoForallPass());
  funcPassManager.addPass(createCanonicalizerPass());

  // Pad the linalg operation
  {
    AMDAIEPadOptions padOptions;
    padOptions.paddingLevel = 1;
    funcPassManager.addPass(createAMDAIEPadPass(padOptions));
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
    tileFuseOptions.tilingLevel = 2;
    tileFuseOptions.useSCFFor = true;
    funcPassManager.addPass(createAMDAIETileAndFusePass(tileFuseOptions));
  }
  funcPassManager.addPass(createAMDAIECleanupPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Pad the linalg operation
  {
    AMDAIEPadOptions padOptions;
    padOptions.paddingLevel = 2;
    funcPassManager.addPass(createAMDAIEPadPass(padOptions));
  }

  // Promote the inputs to local memory
  {
    AMDAIEBufferizeToAllocationOptions bufferizeOptions;
    bufferizeOptions.memorySpace = 2;
    bufferizeOptions.bufferizeOperand = BufferizeOperand::Input;
    funcPassManager.addPass(
        createAMDAIEBufferizeToAllocationPass(bufferizeOptions));
  }

  // Decompose Conv2d ops to Conv1d ops
  funcPassManager.addPass(createDecomposeConvolutionToLowerDimOpsPass());

  // Vectorization passes
  appendVectorizationToPipeline(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());

  // Comprehensive bufferization
  addAMDAIEBufferizePasses(funcPassManager);
}

void buildAMDAIETransformPassPipeline(OpPassManager &variantPassManager,
                                      AMDAIEDevice device) {
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());
  {
    AMDAIELoweringStrategyOptions options;
    options.usePassPipeline = clUseTilePipeline;
    options.useLowerToAIEPipeline = clUseLowerToAIEPipeline;
    modulePassManager.addPass(createAMDAIELoweringStrategyPass(options));
  }
  modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    AMDAIELowerExecutableTargetOptions options;
    options.usePassPipeline = clUseTilePipeline;
    funcPassManager.addPass(
        [&]() { return createAMDAIELowerExecutableTargetPass(options); });
  }
  modulePassManager.addPass(createLowerUKernelOpsToCallsPass());
  if (clUseLowerToAIEPipeline == LowerToAIEPassPipeline::ObjectFifo) {
    addAMDAIEObjectFifoLoweringPasses(modulePassManager);
  } else if (clUseLowerToAIEPipeline == LowerToAIEPassPipeline::AIR) {
    addMLIRAIRLoweringPasses(modulePassManager, device);
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

void addAMDAIEObjectFifoLoweringPasses(OpPassManager &passManager) {
  passManager.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());
  passManager.addPass(createAMDAIEPackToDmaPass());

  passManager.addPass(createAMDAIENormalizeLoopBoundsPass());
  passManager.addPass(createAMDAIEInsertCoresPass());
  passManager.addPass(createAMDAIELocalizeLogicalObjectFifoPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(createAMDAIEDistributeCoresAndObjectFifosPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createAMDAIESplitLogicalObjFifosForConnectionReusePass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIEDmaToCircularDmaPass());
  passManager.addNestedPass<func::FuncOp>(createAMDAIECreateAIEWorkgroupPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createAMDAIEDmaCSEPass());

  passManager.addPass(createAMDAIEHoistLogicalObjFifoPass());
  passManager.addPass(createAMDAIECanonicalizeDoublyStridedOpPass());
  passManager.addPass(createAMDAIEFlattenLogicalObjectFifoPass());
  passManager.addPass(createAMDAIEAssignLogicalObjectFifoDepthPass());
  passManager.addPass(createAMDAIEAccessToAcquireReleasePass());
  passManager.addPass(createAMDAIENoneAccessToTemporaryBufferPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIEDmaCompositionPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createAMDAIEDmaCSEPass());

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

  passManager.addPass(createAMDAIEAssignChannelsPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());

  passManager.addPass(createAMDAIEObjFifoBufferizationPass());

  addAMDAIEToAIEPasses(passManager);

  // Now lower using the AIE passes from MLIR-AIE.
  addMLIRAIELoweringPasses(passManager);
}

// TODO (Erwei): The "packPeel" temporary argument should be removed once
// pack-peel and pack-pad share the same pass pipeline. See TODOs inlined below
// for details.
void addMLIRAIRLoweringPasses(OpPassManager &passManager, AMDAIEDevice device) {
  // Add passes for preparing for lowering to MLIR-AIR
  passManager.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());
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
  if (clUseTilePipeline == TilePassPipeline::ConvDecomposePipeline) {
    passManager.addPass(createCanonicalizerPass());
  }

  passManager.addPass(createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass());

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
  passManager.addPass(xilinx::air::createCopyToDmaPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(xilinx::air::createAIRDependencyPass());
  if (!(clUseTilePipeline == TilePassPipeline::PackPeelPipeline &&
        clMatmulElementwiseFusion)) {
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
  if (clUseTilePipeline != TilePassPipeline::PackPeelPipeline)
    passManager.addNestedPass<func::FuncOp>(
        xilinx::air::createAIRSplitL2MemrefForBufferConstraintPass());
  passManager.addPass(xilinx::air::createAIRIsolateAsyncDmaLoopNests());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  {
    xilinx::air::AIRFuseChannelsOptions options;
    std::vector<std::string> mode;
    if (clUseTilePipeline == TilePassPipeline::PackPeelPipeline &&
        clMatmulElementwiseFusion) {
      mode.push_back("L1");
    }
    options.clAggressiveMode = ArrayRef(mode);
    passManager.addPass(xilinx::air::createAIRFuseChannels(options));
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
    options.clDevice = stringifyEnum(device);
    options.clEmitWhileLoop = true;
    passManager.addPass(xilinx::air::createAIRToAIEPass(options));
  }
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(xilinx::air::createAIRLoweringPass());
  {
    xilinx::air::AffineLoopOptPassOptions options;
    // tile_sizes contains a list of N tiling factors for the N innermost loop
    // nests lowered from the outer scf.forall. The N innermost loops were tiled
    // with given factors, and subsequently unrolled in
    // AIRUnrollOuterPerfectlyNestedLoopsPass, to enforce SHIM DMA BD count
    // within the hardware limit.
    std::vector<unsigned> tile_sizes;
    if (clUseTilePipeline == TilePassPipeline::PackPeelPipeline) {
      tile_sizes = {2, 2};
    } else if (clUseTilePipeline == TilePassPipeline::PadPackPipeline) {
      tile_sizes = {4, 4};
    } else
      tile_sizes = {};
    options.clTileSizes = ArrayRef(tile_sizes);
    passManager.addNestedPass<func::FuncOp>(
        xilinx::air::createAffineLoopOptPass(options));
  }
  passManager.addPass(createCanonicalizerPass());
  {
    // AIRUnrollOuterPerfectlyNestedLoopsPass unrolls the remaining outer loop
    // nests that were left untiled by the previous AffineLoopOptPass,
    // generating NPU sequence representing the SHIM DMA BDs.
    xilinx::air::AIRUnrollOuterPerfectlyNestedLoopsPassOptions options;
    if (clUseTilePipeline == TilePassPipeline::ConvDecomposePipeline)
      options.clDepth = 4;
    else
      options.clDepth = 2;
    passManager.addNestedPass<func::FuncOp>(
        xilinx::air::createAIRUnrollOuterPerfectlyNestedLoopsPass(options));
  }
  passManager.addPass(mlir::affine::createAffineExpandIndexOpsPass());
  passManager.addPass(createAMDAIELowerFuncArgsPass());
  passManager.addPass(xilinx::airrt::createAIRRtToNpuPass());
  passManager.addPass(createCanonicalizerPass());

  // Now lower using the AIE passes from MLIR-AIE.
  addMLIRAIELoweringPasses(passManager);
}

void addMLIRAIELoweringPasses(OpPassManager &passManager) {
  {
    OpPassManager &devicePM = passManager.nest<xilinx::AIE::DeviceOp>();
    devicePM.addPass(createCanonicalizerPass());
    devicePM.addPass(createAMDAIEDmaToNpuPass());
    devicePM.addPass(createAMDAIEAssignBufferDescriptorIDsPass());
    devicePM.addPass(createAMDAIEAssignBufferAddressesBasicPass());
    devicePM.addPass(createAMDAIEPathfinderPass());
  }

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createConvertLinalgToLoopsPass());
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(createConvertSCFToCFPass());

  {
    OpPassManager &devicePM = passManager.nest<xilinx::AIE::DeviceOp>();
    devicePM.addPass(createAMDAIELocalizeLocksPass());
    devicePM.addPass(createAMDAIENormalizeAddressSpacesPass());
    devicePM.addPass(createCanonicalizerPass());
  }
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

void addLowerToLLVMPasses(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(aievec::createConvertAIEVecToLLVMPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  ConvertFuncToLLVMPassOptions opts;
  opts.useBarePtrCallConv = true;
  pm.addPass(createConvertFuncToLLVMPass(opts));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
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
