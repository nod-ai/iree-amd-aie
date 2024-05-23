// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-dma-loop-subsumption"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// bool dependsOnLoop(Value operand) {}

struct RetrieveStrideSize : public AffineExprVisitor<RetrieveStrideSize> {
  std::optional<int64_t> stride;
  RetrieveStrideSize() {}
  void visitMulExpr(AffineBinaryOpExpr expr) {
    if (auto rhsSize = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      stride = rhsSize.getValue();
    } else if (auto lhsSize = dyn_cast<AffineConstantExpr>(expr.getLHS())) {
      stride = lhsSize.getValue();
    }
  }
};

class SubsumeLoopIntoDMA
    : public OpInterfaceRewritePattern<AMDAIE::DoublyStridedOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(AMDAIE::DoublyStridedOpInterface op,
                                PatternRewriter &rewriter) const override {
    // auto strideOp = dyn_cast<AMDAIE::DoublyStridedOpInterface>(op);
    // if (!strideOp) return failure();

    auto forOp = op->getParentOfType<scf::ForOp>();
    if (!forOp) return failure();

    llvm::outs() << "For op: " << forOp << "\n";
    llvm::outs() << "AMDAIE::DoublyStridedOpInterface: " << op << "\n";

    // Only handle loop induction variable with optional `affine.apply`
    // dependency for now.
    Value iv = forOp.getInductionVar();
    affine::AffineApplyOp applyOp;
    for (Operation *userOp : iv.getUsers()) {
      if (auto userApplyOp = dyn_cast<affine::AffineApplyOp>(userOp)) {
        applyOp = userApplyOp;
      }
    }
    if (applyOp && llvm::any_of(op->getOperands(), [&](Value operand) {
          return operand == applyOp.getResult();
        })) {
      llvm::outs() << "Apply op used in operands of strided op\n";   
      AffineMap affineMap = applyOp.getAffineMap();
      llvm::outs() << "Num results: " << affineMap.getNumResults() << "\n";
      llvm::outs() << "Num symbols: " << affineMap.getNumSymbols() << "\n";
      llvm::outs() << "Num inputs: " << affineMap.getNumInputs() << "\n";
      llvm::outs() << "Num dims: " << affineMap.getNumDims() << "\n";
    }

    // RetrieveStrideSize retriever;
    // retriever.visit(applyOp);
    return failure();
  }
};

class AMDAIEDmaLoopSubsumptionPass
    : public impl::AMDAIEDmaLoopSubsumptionBase<AMDAIEDmaLoopSubsumptionPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEDmaLoopSubsumptionPass() = default;
  AMDAIEDmaLoopSubsumptionPass(const AMDAIEDmaLoopSubsumptionPass &pass){};
  void runOnOperation() override;
};

void AMDAIEDmaLoopSubsumptionPass::runOnOperation() {
  // if (failed(subsumeLoopIntoDma(getOperation()))) {
  //   return signalPassFailure();
  // }
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<SubsumeLoopIntoDMA>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDmaLoopSubsumptionPass() {
  return std::make_unique<AMDAIEDmaLoopSubsumptionPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
