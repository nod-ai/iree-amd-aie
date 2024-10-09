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

// class AMDAIEFuseConsumerIntoLoopPass
//     : public impl::AMDAIEFuseConsumerIntoLoopBase<
//           AMDAIEFuseConsumerIntoLoopPass> {
//  public:
//   AMDAIEFuseConsumerIntoLoopPass() = default;
//   AMDAIEFuseConsumerIntoLoopPass(const AMDAIEFuseConsumerIntoLoopPass &pass)
//   {} AMDAIEFuseConsumerIntoLoopPass(
//       const AMDAIEFuseConsumerIntoLoopOptions &options)
//       : AMDAIEFuseConsumerIntoLoopBase(options) {}

//   void getDependentDialects(DialectRegistry &registry) const override {
//     registry.insert<scf::SCFDialect>();
//   }
//   void runOnOperation() override;
// };

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
  // llvm::outs()<<(*moduleOp)<<"\n";

  DenseMap<StringRef, func::FuncOp> outlinedFuncOps;
  auto addFuncDecl = [&](std::string name, FunctionType type) -> func::FuncOp {
    if (moduleOp.lookupSymbol(name)) return outlinedFuncOps[name];
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto funcDeclOp =
        rewriter.create<func::FuncOp>(moduleOp.getLoc(), name, type);
    funcDeclOp.setPrivate();
    outlinedFuncOps[name] = funcDeclOp;
    return funcDeclOp;
  };

  funcOp.walk([&](linalg::LinalgOp op) {
    if (isa<linalg::FillOp, linalg::CopyOp>(op)) return WalkResult::advance();
    std::string computeName = "";
    if (isMatmul(op)) {
      computeName = "_matmul";
    } else {
      // TODO(avarma): Make this better/general.
      computeName = "_elementwise";
    }
    std::string outlinedFuncName =
        op->getName().stripDialect().str() + computeName + "_outlined";
    SmallVector<Type> inputTypes = llvm::map_to_vector(
        op.getDpsInputs(), [](Value v) { return v.getType(); });
    SmallVector<Type> outputTypes =
        llvm::map_to_vector(op.getDpsInits(), [&](Value v) {
          inputTypes.push_back(v.getType());
          return v.getType();
        });
    func::FuncOp funcDeclOp = addFuncDecl(
        outlinedFuncName,
        FunctionType::get(rewriter.getContext(), inputTypes, outputTypes));
    // if (!funcDeclOp) return WalkResult::advance();
    rewriter.setInsertionPoint(op);
    auto callOp = rewriter.create<func::CallOp>(op.getLoc(), funcDeclOp,
                                                op->getOperands());
    rewriter.replaceOp(op, callOp.getResults());
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFunctionOutliningPass() {
  return std::make_unique<AMDAIEFunctionOutliningPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
