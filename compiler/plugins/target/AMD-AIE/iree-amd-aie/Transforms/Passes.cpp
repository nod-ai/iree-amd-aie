// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "air/Conversion/Passes.h"
#include "air/Transform/Passes.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {
#define GEN_PASS_DECL
#include "air/Conversion/Passes.h.inc"
}  // namespace
namespace {
#define GEN_PASS_DECL
#include "air/Transform/Passes.h.inc"
}  // namespace

void buildAMDAIETransformPassPipeline(OpPassManager &pm) {
  addCommonTargetExecutablePreprocessingPasses(pm);
  pm.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  pm.addPass(createAMDAIELowerExecutableTargetPass());
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
    ParallelToHerdOptions options;
    options.clAssignDepth = 1;
    passManager.addPass(xilinx::air::createParallelToHerdPass()); // depth=1
  }
  {
    ParallelToLaunchOptions options;
    options.clHasSegment = true;
    passManager.addPass(xilinx::air::createParallelToLaunchPass()); // has-air-segment=true
  }
  passManager.addPass(xilinx::air::createCopyToDmaPass());

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(xilinx::air::createAIRDependencyPass());
  passManager.addPass(xilinx::air::createAIRDependencyScheduleOptPass());
  passManager.addPass(xilinx::air::createDmaToChannelPass());

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(xilinx::air::createAIRDependencyCanonicalizePass());

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(xilinx::air::createAIRLabelScfForLoopForPingPongPattern());
  {
    AIRPingPongTransformationPatternOptions options;
    options.clKeepMemrefDealloc = true;
    passManager.addPass(xilinx::air::createAIRPingPongTransformationPattern()); // keep-memref-dealloc=true
  }
  passManager.addPass(xilinx::air::createAIRDeAliasMemref());

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(xilinx::air::createAIRLabelScfForLoopInAIRSegmentPattern());
  passManager.addPass(xilinx::air::createAIRUnrollLoopForPipeliningPattern());
  {
    AIRHerdPlacementPassOptions options;
    options.clNumRows = 2;
    options.clNumCols = 2;
    options.clAnchorPointRow = 2;
    options.clAnchorPointCol = 0;
    passManager.addPass(xilinx::air::createAIRHerdPlacementPass()); // num-rows=2 num-cols=2 row-anchor=2 col-anchor=0
  }

  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addNestedPass<func::FuncOp>(xilinx::air::createAIRRenumberDmaIdPass());
  passManager.addNestedPass<func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
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
