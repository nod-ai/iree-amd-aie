// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"

#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "air/Conversion/Passes.h"
#include "air/Transform/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/CPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
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
        clEnumValN(AIEPassPipeline::PadPipeline, "pad",
                   "Use IREE lowering to AIR dialect through pad operations"),
        clEnumValN(
            AIEPassPipeline::PackPipeline, "pack",
            "Use the IREE lowering to AIR dialect through pack operation"),
        clEnumValN(AIEPassPipeline::SimplePackPipeline, "simple-pack",
                   "Use the simplified IREE lowering to AIR dialect through "
                   "pack operation")),
    llvm::cl::init(AIEPassPipeline::SimplePackPipeline));

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

void addPadBasedPassPipeline(OpPassManager &pm, TilingConfig &tilingConfig) {
  auto &modulePassManager = pm.nest<ModuleOp>();
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  int64_t memorySpace = 1;
  for (unsigned i = 0, n = tilingConfig.getNumTilingLevels(); i < n; i++) {
    {
      AMDAIETileAndFuseOptions options;
      if (i == 2) {
        options.useSCFFor = true;
      }
      options.tilingLevel = i;
      modulePassManager.addNestedPass<func::FuncOp>(
          createAMDAIETileAndFusePass(options));
    }
    if (i == 2) {
      modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
      pm.addPass(createCanonicalizerPass());
      pm.addPass(createCSEPass());
    }
    {
      AMDAIEPadOptions options;
      options.paddingLevel = i;
      modulePassManager.addNestedPass<func::FuncOp>(
          createAMDAIEPadPass(options));
    }
    {
      AMDAIEBufferizeToAllocationOptions options;
      if (i == 1) {
        memorySpace = 2;
      }
      options.memorySpace = memorySpace;
      options.bufferizeLevel = i;
      modulePassManager.addNestedPass<func::FuncOp>(
          createAMDAIEBufferizeToAllocationPass(options));
    }
    modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
  addAMDAIEBufferizePasses(modulePassManager);
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void addSimplePackBasedPassPipeline(OpPassManager &pm,
                                    TilingConfig &tilingConfig) {
  auto &modulePassManager = pm.nest<ModuleOp>();
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  AMDAIETileAndFuseOptions tileOptions;
  AMDAIEPackAndTransposeOptions packOptions;
  AMDAIEBufferizeToAllocationOptions bufferizeOptions;

  // First level tiling using scf.forall
  tileOptions.tilingLevel = 0;
  tileOptions.useSCFFor = false;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIETileAndFusePass(tileOptions));
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // First level packing and bufferize to allocation
  packOptions.packLevel = 0;
  packOptions.usePassPipeline = AIEPassPipeline::SimplePackPipeline;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIEPackAndTransposePass(packOptions));
  bufferizeOptions.memorySpace = 1;
  bufferizeOptions.bufferizeLevel = -1;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIEBufferizeToAllocationPass(bufferizeOptions));

  // Second level tiling using scf.forall
  tileOptions.tilingLevel = 1;
  tileOptions.useSCFFor = false;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIETileAndFusePass(tileOptions));
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // Fuse fill into forall loop
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIEFuseFillIntoForallPass());
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // Second level packing and bufferize to allocation
  packOptions.packLevel = 1;
  packOptions.usePassPipeline = AIEPassPipeline::SimplePackPipeline;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIEPackAndTransposePass(packOptions));
  bufferizeOptions.memorySpace = 2;
  bufferizeOptions.bufferizeLevel = -1;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIEBufferizeToAllocationPass(bufferizeOptions));

  // Tile the reduction loops
  tileOptions.tilingLevel = 2;
  tileOptions.useSCFFor = true;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIETileAndFusePass(tileOptions));
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // Comprehensive bufferization
  addAMDAIEBufferizePasses(modulePassManager);
}

void addPackBasedPassPipeline(OpPassManager &pm, TilingConfig &tilingConfig) {
  auto &modulePassManager = pm.nest<ModuleOp>();
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  AMDAIETileAndFuseOptions tileOptions;
  AMDAIEPackAndTransposeOptions packOptions;
  AMDAIEBufferizeToAllocationOptions bufferizeOptions;

  // First level tiling using scf.forall
  tileOptions.tilingLevel = 0;
  tileOptions.useSCFFor = false;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIETileAndFusePass(tileOptions));
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // Tile the reduction loops
  tileOptions.tilingLevel = 1;
  tileOptions.useSCFFor = true;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIETileAndFusePass(tileOptions));
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // First level packing and bufferize to allocation
  packOptions.packLevel = 0;
  packOptions.usePassPipeline = AIEPassPipeline::PackPipeline;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIEPackAndTransposePass(packOptions));
  bufferizeOptions.memorySpace = 1;
  bufferizeOptions.bufferizeLevel = -1;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIEBufferizeToAllocationPass(bufferizeOptions));

  // Second level tiling using scf.forall
  tileOptions.tilingLevel = 2;
  tileOptions.useSCFFor = false;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIETileAndFusePass(tileOptions));
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIECleanupPass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // Second level packing and bufferize to allocation
  packOptions.packLevel = 1;
  packOptions.usePassPipeline = AIEPassPipeline::PackPipeline;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIEPackAndTransposePass(packOptions));
  bufferizeOptions.memorySpace = 2;
  bufferizeOptions.bufferizeLevel = -1;
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIEBufferizeToAllocationPass(bufferizeOptions));

  // Hoist static allocations
  modulePassManager.addNestedPass<func::FuncOp>(
      createHoistStaticallyBoundAllocationsPass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // Peel the first iteration out of the for loop
  modulePassManager.addNestedPass<func::FuncOp>(createAMDAIEPeelForLoopPass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // Fuse fill into forall loop
  modulePassManager.addNestedPass<func::FuncOp>(
      createAMDAIEFuseFillIntoForallPass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  // Comprehensive bufferization
  addAMDAIEBufferizePasses(modulePassManager);
}

void buildAMDAIETransformPassPipeline(OpPassManager &pm) {
  addCommonTargetExecutablePreprocessingPasses(pm);
  {
    AMDAIELoweringStrategyOptions options;
    options.usePassPipeline = clUsePipeline;
    pm.addPass(createAMDAIELoweringStrategyPass(options));
  }
  {
    AMDAIELowerExecutableTargetOptions options;
    options.usePassPipeline = clUsePipeline;
    pm.addPass(createAMDAIELowerExecutableTargetPass(options));
  }
  pm.addPass(createAMDAIELowerWorkgroupCountPass());

  auto &modulePassManager = pm.nest<ModuleOp>();
  addMLIRAIRAIELoweringPasses(modulePassManager);

  LLVM_DEBUG({
    llvm::dbgs() << "Using AMDAIE pass pipeline:\n";
    pm.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

void addTransformDialectPasses(OpPassManager &passManager) {
  // Give control to the transform dialect.
  passManager.addPass(
      mlir::iree_compiler::createTransformDialectInterpreterPass());
  // Dropping the schedule is needed:
  //   1. if we want to embed the transform in the module: we should drop the
  //      schedule once applied.
  //   2. if transform.do_not_dce_operands ops are introduced.
  passManager.addPass(createDropSchedulePass());
}

void addMLIRAIRAIELoweringPasses(OpPassManager &passManager) {
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

  passManager.addPass(
      xilinx::air::createAIRLabelScfForLoopForPingPongPattern());
  {
    xilinx::air::AIRPingPongTransformationPatternOptions options;
    options.clKeepMemrefDealloc = true;
    passManager.addPass(
        xilinx::air::createAIRPingPongTransformationPattern(options));
  }
  passManager.addPass(xilinx::air::createAIRDeAliasMemref());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(xilinx::air::createAIRIsolateAsyncDmaLoopNests());
  passManager.addPass(
      xilinx::air::createAIRSpecializeChannelWrapAndStridePattern());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addNestedPass<func::FuncOp>(
      xilinx::air::createAIRCollapseHerdPass());
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
  passManager.addPass(xilinx::air::createAIRLoweringPass());
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
        addMLIRAIRAIELoweringPasses(passManager);
      });
}

}  // namespace mlir::iree_compiler::AMDAIE
