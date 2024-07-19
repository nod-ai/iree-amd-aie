// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering from the AMDAIE dialect to AIE and AIEX
// dialects.
//
//===----------------------------------------------------------------------===//

#include <numeric>

#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-lower-func-args"

namespace mlir::iree_compiler::AMDAIE {

namespace {

//===----------------------------------------------------------------------===//
// Convert the module operation's contents to the AIE dialect
//===----------------------------------------------------------------------===//

LogicalResult lowerFuncArgs(ModuleOp moduleOp) {
  auto funcRes = moduleOp.walk([](func::FuncOp funcOp) {
    if (funcOp.isPrivate()) {
      return WalkResult::advance();
    }
    SmallVector<IREE::HAL::InterfaceBindingSubspanOp> subspanOps;
    funcOp->walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
      subspanOps.push_back(subspanOp);
    });
    llvm::sort(subspanOps, [](IREE::HAL::InterfaceBindingSubspanOp a,
                              IREE::HAL::InterfaceBindingSubspanOp b) {
      return a.getBinding().getZExtValue() < b.getBinding().getZExtValue();
    });
    SmallVector<Type> inputTypes;
    for (auto op : subspanOps) inputTypes.push_back(op.getType());

    auto functionType = funcOp.getFunctionType();
    auto newArgTypes = llvm::to_vector<6>(
        llvm::concat<const Type>(functionType.getInputs(), inputTypes));
    auto newFunctionType = FunctionType::get(funcOp.getContext(), newArgTypes,
                                             functionType.getResults());
    funcOp.setType(newFunctionType);
    for (int i = 0; i < subspanOps.size(); ++i) {
      auto newArg =
          funcOp.front().addArgument(inputTypes[i], subspanOps[i]->getLoc());
      subspanOps[i].replaceAllUsesWith(newArg);
    }
    LLVM_DEBUG(llvm::dbgs() << "function after lowerFuncArgs: " << funcOp);
    return WalkResult::advance();
  });
  if (funcRes.wasInterrupted()) return failure();
  return success();
}

class AMDAIELowerFuncArgsPass
    : public impl::AMDAIELowerFuncArgsBase<AMDAIELowerFuncArgsPass> {
 public:
  AMDAIELowerFuncArgsPass() = default;
  AMDAIELowerFuncArgsPass(const AMDAIELowerFuncArgsPass &pass){};
  void runOnOperation() override;
};

void AMDAIELowerFuncArgsPass::runOnOperation() {
  if (failed(lowerFuncArgs(getOperation()))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELowerFuncArgsPass() {
  return std::make_unique<AMDAIELowerFuncArgsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
