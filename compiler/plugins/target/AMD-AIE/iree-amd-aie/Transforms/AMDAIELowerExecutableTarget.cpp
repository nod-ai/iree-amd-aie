// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/KernelDispatch.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Lowers an hal.executable.variant operation to scalar/native-vector code.
/// This should be merged with the equivalent pass in LinalgToLLVM. For
/// simplicity it is currently a separate pass.
class AMDAIELowerExecutableTargetPass
    : public impl::AMDAIELowerExecutableTargetBase<
          AMDAIELowerExecutableTargetPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        IREE::HAL::HALDialect, IREE::LinalgExt::IREELinalgExtDialect,
        bufferization::BufferizationDialect, linalg::LinalgDialect,
        LLVM::LLVMDialect, pdl::PDLDialect, pdl_interp::PDLInterpDialect,
        scf::SCFDialect, tensor::TensorDialect, transform::TransformDialect,
        vector::VectorDialect>();
  }

  AMDAIELowerExecutableTargetPass() = default;
  AMDAIELowerExecutableTargetPass(
      const AMDAIELowerExecutableTargetPass &pass){};
  AMDAIELowerExecutableTargetPass(
      const AMDAIELowerExecutableTargetOptions &options)
      : AMDAIELowerExecutableTargetBase(options) {}

  void runOnOperation() override;
};
}  // namespace

static Operation *getRootOp(FunctionOpInterface funcOp) {
  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  assert(succeeded(rootOp) && "Pipeline requires a root operation");
  return rootOp.value();
}

void AMDAIELowerExecutableTargetPass::runOnOperation() {
  auto funcOp = getOperation();
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!target) {
    // Do nothing without target
    return;
  }

  IREE::Codegen::TranslationInfoAttr translationInfo =
      getTranslationInfo(funcOp);
  if (!translationInfo) return;

  OpPassManager executableLoweringPipeline(func::FuncOp::getOperationName());
  switch (translationInfo.getDispatchLoweringPassPipeline()) {
      // No pipleline specified, nothing to do.
    case IREE::Codegen::DispatchLoweringPassPipeline::None:
      return;
    case IREE::Codegen::DispatchLoweringPassPipeline::Custom: {
      if (useTilePipeline == TilePassPipeline::PackPeelPipeline) {
        addPackPeelBasedPassPipeline(executableLoweringPipeline,
                                     TilePassPipeline::PackPeelPipeline);
      } else if (useTilePipeline ==
                 TilePassPipeline::PackPeel4LevelTilingPipeline) {
        addPackPeel4LevelTilingBasedPassPipeline(
            executableLoweringPipeline,
            TilePassPipeline::PackPeel4LevelTilingPipeline, getRootOp(funcOp));
      } else if (useTilePipeline == TilePassPipeline::ConvDecomposePipeline) {
        addConvDecomposePassPipeline(executableLoweringPipeline,
                                     TilePassPipeline::ConvDecomposePipeline);
      } else if (useTilePipeline == TilePassPipeline::GeneralCopyPipeline) {
        addGeneralCopyPassPipeline(executableLoweringPipeline,
                                   TilePassPipeline::GeneralCopyPipeline,
                                   getRootOp(funcOp));
      }
      break;
    }
    default:
      funcOp.emitOpError("unhandled pass pipeline value set");
      return signalPassFailure();
  }

  if (failed(runPipeline(executableLoweringPipeline, funcOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAMDAIELowerExecutableTargetPass(
    AMDAIELowerExecutableTargetOptions options) {
  return std::make_unique<AMDAIELowerExecutableTargetPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
