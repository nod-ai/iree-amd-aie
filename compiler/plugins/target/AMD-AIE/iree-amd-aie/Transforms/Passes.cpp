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
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

/// Command line options used purely for development purposes. Not to be relied
/// on in any way.
static llvm::cl::opt<bool> clUseCPlusPlusTransformPasses(
    "iree-amd-aie-cpp-passes",
    llvm::cl::desc(
        "Runs the cpp passes instead of transform dialect when possible"),
    llvm::cl::init(true));

namespace mlir::iree_compiler::AMDAIE {

void buildAMDAIETransformPassPipeline(OpPassManager &pm) {
  addCommonTargetExecutablePreprocessingPasses(pm);
  // TODO: Current we don't include C++ equivalent of Transform dialect scripts.
  // We are thus guarding their inclusion with a bool
  // `useCPlusPlusTransformPasses` which would be used during development
  // efforts. Once the C++ passes are ready, we will include these passes by
  // default and take away the guarding.
  if (clUseCPlusPlusTransformPasses) {
    pm.addPass(createCleanupPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createAMDAIETileAndFusePass(1));
    pm.addPass(createAMDAIEPadAndBufferizePass(1));
    pm.addPass(createCleanupPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createAMDAIETileAndFusePass(2));
    pm.addPass(createAMDAIEPadAndBufferizePass(2));
    pm.addPass(createCleanupPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createAMDAIETileAndFusePass(3));
    pm.addPass(createCleanupPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createAMDAIEPadAndBufferizePass(3));
    pm.addPass(createCleanupPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
  pm.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  pm.addPass(createAMDAIELowerExecutableTargetPass());

  auto &modulePassManager = pm.nest<ModuleOp>();
  addMLIRAIRAIELoweringPasses(modulePassManager);
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
  passManager.addPass(createAMDAIEBridgeToAIRPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());

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
  passManager.addPass(xilinx::air::createAIRSpecializeChannelWrapAndStridePattern());
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
