// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/KernelDispatch.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {
/// Add the lowering strategy configurations to be used for ops.
class AMDAIELoweringStrategyPass
    : public impl::AMDAIELoweringStrategyBase<AMDAIELoweringStrategyPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        AMDAIE::AMDAIEDialect, IREE::Codegen::IREECodegenDialect,
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
  ModuleOp moduleOp = getOperation();

  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    // Set the strategy with default heuristics.
    if (failed(initAIELaunchConfig(funcOp, useTilePipeline,
                                   useLowerToAIEPipeline, targetDevice, numRows,
                                   numCols, stackSize, enableAMDAIEUkernels))) {
      funcOp.emitOpError("failed to have a lowering configuration set for it.");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createAMDAIELoweringStrategyPass(
    AMDAIELoweringStrategyOptions options) {
  return std::make_unique<AMDAIELoweringStrategyPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
