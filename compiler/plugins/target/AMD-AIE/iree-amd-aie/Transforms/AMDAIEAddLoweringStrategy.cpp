// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/KernelDispatch.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {
/// Add the lowering strategy configurations to be used for ops.
class AMDAIELoweringStrategyPass
    : public impl::AMDAIELoweringStrategyBase<AMDAIELoweringStrategyPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        IREE::HAL::HALDialect, IREE::LinalgExt::IREELinalgExtDialect,
        bufferization::BufferizationDialect, linalg::LinalgDialect,
        LLVM::LLVMDialect, pdl::PDLDialect, pdl_interp::PDLInterpDialect,
        scf::SCFDialect, tensor::TensorDialect, transform::TransformDialect,
        vector::VectorDialect>();
  }

  AMDAIELoweringStrategyPass() = default;
  AMDAIELoweringStrategyPass(const AMDAIELoweringStrategyOptions &options)
      : AMDAIELoweringStrategyBase(options) {}
  AMDAIELoweringStrategyPass(const AMDAIELoweringStrategyPass &pass){};

  void runOnOperation() override;
};
}  // namespace

void AMDAIELoweringStrategyPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  ModuleOp moduleOp = variantOp.getInnerModule();
  if (!moduleOp) {
    getOperation()->emitError(
        "Expected a variantOp root with an inner ModuleOp");
    return signalPassFailure();
  }
  if (failed(initAIELaunchConfig(moduleOp, usePassPipeline, useMulticore))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createAMDAIELoweringStrategyPass(
    AMDAIELoweringStrategyOptions options) {
  return std::make_unique<AMDAIELoweringStrategyPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
