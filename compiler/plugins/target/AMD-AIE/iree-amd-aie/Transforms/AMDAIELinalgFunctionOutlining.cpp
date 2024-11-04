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

#define DEBUG_TYPE "iree-amdaie-linalg-function-outlining"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Utility to outline the linalg compute op.
static FailureOr<func::FuncOp> outlinedToAFunction(
    IRRewriter &rewriter, ModuleOp moduleOp, linalg::LinalgOp computeOp,
    std::string outlineFuncName) {
  if (auto outlinedFuncOp = dyn_cast_if_present<func::FuncOp>(
          moduleOp.lookupSymbol(outlineFuncName))) {
    return outlinedFuncOp;
  }

  // Form outlined FunctionType.
  SmallVector<Type> inputTypes = llvm::map_to_vector(
      computeOp.getDpsInputs(), [](Value v) { return v.getType(); });
  for (Value val : computeOp.getDpsInits()) inputTypes.push_back(val.getType());
  auto outlinedFuncType =
      FunctionType::get(rewriter.getContext(), inputTypes, /*outputTypes=*/{});

  // Form outlined FuncSignature
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto outlinedFunc = rewriter.create<func::FuncOp>(
      moduleOp.getLoc(), outlineFuncName, outlinedFuncType);
  outlinedFunc.setPrivate();

  // Create an entry func block and map the original operands of the compute
  // op to the block arguments.
  Block *outlinedFuncBody = outlinedFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(outlinedFuncBody);
  SmallVector<BlockArgument> outlinedFuncArgs = llvm::map_to_vector(
      outlinedFunc.getArguments(), [&](BlockArgument bbArg) { return bbArg; });
  unsigned bbArgIndex = 0;
  IRMapping operandMap;
  for (Value origOperand : computeOp.getDpsInputs())
    operandMap.map(origOperand, outlinedFuncArgs[bbArgIndex++]);
  for (Value origOperand : computeOp.getDpsInits())
    operandMap.map(origOperand, outlinedFuncArgs[bbArgIndex++]);

  // Clone the compute op while mapping the operand to the function block
  // arguments.
  Operation *clonedComputeOp = rewriter.clone(*computeOp, operandMap);

  // Create terminator op returning the cloned compute op's results.
  rewriter.setInsertionPointToEnd(outlinedFuncBody);
  rewriter.create<func::ReturnOp>(clonedComputeOp->getLoc(), ValueRange({}));

  return outlinedFunc;
}

class AMDAIELinalgFunctionOutliningPass
    : public impl::AMDAIELinalgFunctionOutliningBase<
          AMDAIELinalgFunctionOutliningPass> {
 public:
  AMDAIELinalgFunctionOutliningPass() = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override;
};

void AMDAIELinalgFunctionOutliningPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  unsigned uniqueOutlinedMatmul = 0;
  unsigned uniqueOutlinedElementwise = 0;
  DenseMap<Operation *, std::string> computeOpToOutlinedFuncMap;
  SmallVector<Operation *> toBeErased;
  moduleOp.walk([&](linalg::LinalgOp computeOp) {
    // Form outlined function name for matmul/elementwise compute ops.
    std::string outlineFuncName = "";
    // Check if the compute op is equivalent to a previously outlined compute
    // op. If yes, we replace the `outlineFuncName` of the current compute op to
    // be same as the previous equivalent outlined compute op in order to lookup
    // the Symbol table.
    for (auto &[op, funcName] : computeOpToOutlinedFuncMap) {
      if (OperationEquivalence::isEquivalentTo(
              computeOp.getOperation(), op,
              OperationEquivalence::ignoreValueEquivalence, /*flags=*/nullptr,
              OperationEquivalence::IgnoreLocations)) {
        outlineFuncName = funcName;
        break;
      }
    }
    if (outlineFuncName == "") {
      std::string computeName = "";
      if (isMatmul(computeOp)) {
        computeName = "_matmul_" + std::to_string(uniqueOutlinedMatmul++);
      } else if (isElementwise(computeOp)) {
        computeName =
            "_elementwise_" + std::to_string(uniqueOutlinedElementwise++);
      } else {
        computeOp->emitRemark()
            << "support to outline this linalg op is missing";
        return WalkResult::skip();
      }
      outlineFuncName =
          computeOp->getName().stripDialect().str() + computeName + "_outlined";
      computeOpToOutlinedFuncMap[computeOp] = outlineFuncName;
    }

    FailureOr<func::FuncOp> outlinedFuncOp =
        outlinedToAFunction(rewriter, moduleOp, computeOp, outlineFuncName);
    if (failed(outlinedFuncOp)) return WalkResult::interrupt();
    rewriter.setInsertionPoint(computeOp);
    rewriter.create<func::CallOp>(computeOp.getLoc(), *outlinedFuncOp,
                                  computeOp->getOperands());
    // We cannot immediately erase the compute op because it'd be used for
    // equivalence check.
    toBeErased.push_back(computeOp);
    return WalkResult::advance();
  });
  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELinalgFunctionOutliningPass() {
  return std::make_unique<AMDAIELinalgFunctionOutliningPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
