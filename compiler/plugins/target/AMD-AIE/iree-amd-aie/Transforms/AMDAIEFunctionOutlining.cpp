// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-amdaie-function-outlining"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEFunctionOutliningPass
    : public impl::AMDAIEFunctionOutliningBase<AMDAIEFunctionOutliningPass> {
 public:
  AMDAIEFunctionOutliningPass() = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEFunctionOutliningPass::runOnOperation() {
  mlir::FunctionOpInterface funcOp = getOperation();
  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  auto outlinedToAFunction = [&](linalg::LinalgOp &computeOp) -> func::FuncOp {
    // Form outlined FuncName.
    std::string computeName = "";
    if (isMatmul(computeOp)) {
      computeName = "_matmul";
    } else {
      // TODO(avarma): Make this better/general.
      computeName = "_elementwise";
    }
    std::string outlinedFuncName =
        computeOp->getName().stripDialect().str() + computeName + "_outlined";
    if (auto outlinedFuncOp = dyn_cast_if_present<func::FuncOp>(
            moduleOp.lookupSymbol(outlinedFuncName)))
      return outlinedFuncOp;

    // Form outlined FunctionType.
    SmallVector<Type> inputTypes = llvm::map_to_vector(
        computeOp.getDpsInputs(), [](Value v) { return v.getType(); });
    SmallVector<Type> outputTypes =
        llvm::map_to_vector(computeOp.getDpsInits(), [&](Value v) {
          inputTypes.push_back(v.getType());
          return v.getType();
        });
    auto outlinedFuncType =
        FunctionType::get(rewriter.getContext(), inputTypes, outputTypes);

    // Form outlined FuncSignature
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto outlinedFunc = rewriter.create<func::FuncOp>(
        moduleOp.getLoc(), outlinedFuncName, outlinedFuncType);
    outlinedFunc.setPrivate();

    // Create an entry func block and map the original operands of the compute
    // op to the block arguments.
    Block *outlinedFuncBody = outlinedFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(outlinedFuncBody);
    SmallVector<BlockArgument> outlinedFuncArgs =
        llvm::map_to_vector(outlinedFunc.getArguments(),
                            [&](BlockArgument bbArg) { return bbArg; });
    unsigned bbArgIndex = 0;
    IRMapping operandMap;
    for (Value origOperand : computeOp.getDpsInputs()) {
      operandMap.map(origOperand, outlinedFuncArgs[bbArgIndex++]);
    }
    for (Value origOperand : computeOp.getDpsInits()) {
      operandMap.map(origOperand, outlinedFuncArgs[bbArgIndex++]);
    }

    // Clone the compute op while mapping the operand to the function block
    // arguments.
    Operation *clonedComputeOp = rewriter.clone(*computeOp, operandMap);

    // Create terminator op returning the cloned compute op's results.
    rewriter.setInsertionPointToEnd(outlinedFuncBody);
    rewriter.create<func::ReturnOp>(clonedComputeOp->getLoc(),
                                    clonedComputeOp->getResult(0));

    return outlinedFunc;
  };

  funcOp.walk([&](linalg::LinalgOp computeOp) {
    if (isa<linalg::FillOp, linalg::CopyOp>(computeOp))
      return WalkResult::skip();
    func::FuncOp outlinedFuncOp = outlinedToAFunction(computeOp);
    rewriter.setInsertionPoint(computeOp);
    auto callOp = rewriter.create<func::CallOp>(
        computeOp.getLoc(), outlinedFuncOp, computeOp->getOperands());
    rewriter.replaceOp(computeOp, callOp.getResults());
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFunctionOutliningPass() {
  return std::make_unique<AMDAIEFunctionOutliningPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
