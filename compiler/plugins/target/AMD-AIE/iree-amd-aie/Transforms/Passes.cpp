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
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::AMDAIE {

void buildAMDAIETransformPassPipeline(OpPassManager &pm) {
  addCommonTargetExecutablePreprocessingPasses(pm);
  pm.addPass(createEraseHALDescriptorTypeFromMemRefPass());
  pm.addPass(createAMDAIELowerExecutableTargetPass());

  auto &modulePassManager = pm.nest<ModuleOp>();
  addMLIRAIRAIELoweringPasses(modulePassManager);
  addMLIRAIELoweringPasses(modulePassManager);
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

  passManager.addPass(xilinx::air::createAIRFuseChannels());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  passManager.addPass(
      xilinx::air::createAIRLabelScfForLoopInAIRSegmentPattern());
  passManager.addPass(xilinx::air::createAIRUnrollLoopForPipeliningPattern());

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
    passManager.addPass(xilinx::air::createAIRToAIEPass(options));
  }
  passManager.addPass(xilinx::air::createAIRLoweringPass());
  passManager.addPass(xilinx::airrt::createAIRRtToIpuPass());
  passManager.addPass(createCanonicalizerPass());
}

void addMLIRAIELoweringPasses(OpPassManager &passManager) {
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(xilinx::AIE::createAIECanonicalizeDevicePass());

  {
    OpPassManager &devicePassManager =
        passManager.nest<xilinx::AIE::DeviceOp>();
    devicePassManager.addPass(xilinx::AIE::createAIEAssignLockIDsPass());
    devicePassManager.addPass(
        xilinx::AIE::createAIEObjectFifoRegisterProcessPass());
    devicePassManager.addPass(
        xilinx::AIE::createAIEObjectFifoStatefulTransformPass());
    devicePassManager.addPass(xilinx::AIEX::createAIEBroadcastPacketPass());
    devicePassManager.addPass(xilinx::AIE::createAIERoutePacketFlowsPass());
    devicePassManager.addPass(xilinx::AIEX::createAIELowerMulticastPass());
    devicePassManager.addPass(
        xilinx::AIE::createAIEAssignBufferAddressesPass());
  }

  // Convert to LLVM.
  passManager.addPass(createConvertSCFToCFPass());
  {
    OpPassManager &devicePassManager =
        passManager.nest<xilinx::AIE::DeviceOp>();
    devicePassManager.addPass(xilinx::AIE::createAIELocalizeLocksPass());
  }
  passManager.addPass(xilinx::AIE::createAIECoreToStandardPass());
  passManager.addPass(xilinx::AIEX::createAIEXToStandardPass());
  passManager.addNestedPass<xilinx::AIE::DeviceOp>(
      xilinx::AIE::createAIENormalizeAddressSpacesPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createConvertVectorToLLVMPass());
  passManager.addPass(memref::createExpandStridedMetadataPass());
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(createConvertMathToLLVMPass());
  passManager.addPass(createArithToLLVMConversionPass());
  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  {
    ConvertFuncToLLVMPassOptions options;
    options.useBarePtrCallConv = true;
    passManager.addPass(createConvertFuncToLLVMPass(options));
  }
  passManager.addPass(createConvertControlFlowToLLVMPass());
  passManager.addPass(createReconcileUnrealizedCastsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
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
