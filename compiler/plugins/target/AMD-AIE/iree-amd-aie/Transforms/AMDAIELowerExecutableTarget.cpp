// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/KernelDispatch.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
// TODO(avarma):
//     We shouldn't add "CPUUtils" - instead it should perhaps just be "Utils"?
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"

using mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr;

namespace mlir::iree_compiler::AMDAIE {

namespace {
/// Lowers an hal.executable.variant operation to scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code
/// - then convert to NVVM/ROCDL dialect.
/// This should be merged with the equivalent pass in LinalgToLLVM. Fo
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
      const AMDAIELowerExecutableTargetOptions &options)
      : AMDAIELowerExecutableTargetBase(options) {}
  AMDAIELowerExecutableTargetPass(
      const AMDAIELowerExecutableTargetPass &pass){};

  void runOnOperation() override;
};
}  // namespace

// TODO(dcaballe): We temporarily need this utility to retrieve a valid
// lowering config. We should be able to remove this once we have a lowering
// config attribute per op.
static FailureOr<LoweringConfigAttr> getRootLoweringConfig(ModuleOp moduleOp) {
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  for (auto &it : exportOps) {
    auto exportOp = it.second;
    auto rootLoweringConfig = iree_compiler::getLoweringConfig(exportOp);
    if (rootLoweringConfig) {
      return rootLoweringConfig;
    }
  }

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    getAllEntryPoints(moduleOp);
    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    // Check for self first.
    FailureOr<Operation *> rootOp = getRootOperation(computeOps);
    auto rootLoweringConfig = iree_compiler::getLoweringConfig(rootOp.value());
    if (rootLoweringConfig) {
      return rootLoweringConfig;
    }
  }

  return failure();
}

static TilingConfig getTilingConfigForPipeline(ModuleOp moduleOp) {
  auto maybeLoweringConfig = getRootLoweringConfig(moduleOp);
  assert(succeeded(maybeLoweringConfig) &&
         "Pipeline requires a lowering config");
  return TilingConfig(*maybeLoweringConfig);
}

void AMDAIELowerExecutableTargetPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();
  if (!moduleOp) {
    getOperation()->emitError(
        "Expected a variantOp root with an inner ModuleOp");
    return signalPassFailure();
  }
  // TODO (nmeshram): ADD a LoweringStrategy pass where this should be moved and
  // then the lowering startegy should be verified
  if (failed(initAIELaunchConfig(moduleOp, tilingStrategy))) {
    return signalPassFailure();
  }

  OpPassManager executableLoweringPipeline(
      IREE::HAL::ExecutableVariantOp::getOperationName());

  // There might be multiple entry points in the module. Currently, all of
  // them need to have the same translation info. This should already be
  // verified when the strategies are set, but we still need to retrieve the
  // correct translation info.
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo;
  for (auto &it : exportOps) {
    auto exportOp = it.second;
    if (IREE::Codegen::TranslationInfoAttr currTranslationInfo =
            getTranslationInfo(exportOp)) {
      if (translationInfo) {
        if (currTranslationInfo != translationInfo.value()) {
          moduleOp.emitOpError(
              "unhandled compilation of entry point functions with different "
              "translation info");
          return signalPassFailure();
        }
      } else {
        translationInfo = currTranslationInfo;
      }
    }
  }

  if (translationInfo.has_value()) {
    switch (translationInfo.value().getDispatchLoweringPassPipeline()) {
      // Transform-dialect pipelines.
      case IREE::Codegen::DispatchLoweringPassPipeline::TransformDialectCodegen:
        addTransformDialectPasses(executableLoweringPipeline);
        break;
      // TODO(avarma): Currently we are using "CPUDefault" but resorting to use
      //               the default case. Will soon have corresponding AIE enum.
      default:
        TilingConfig tilingConfig = getTilingConfigForPipeline(moduleOp);
        addPadBasedPassPipeline(executableLoweringPipeline, tilingConfig);
        break;
    }
  }

  if (failed(runPipeline(executableLoweringPipeline, variantOp))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createAMDAIELowerExecutableTargetPass(
    AMDAIELowerExecutableTargetOptions options) {
  return std::make_unique<AMDAIELowerExecutableTargetPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
