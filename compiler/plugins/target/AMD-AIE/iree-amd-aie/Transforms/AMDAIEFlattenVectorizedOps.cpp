// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"

#define DEBUG_TYPE "iree-amdaie-flatten-vectorized-ops"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEFlattenVectorizedOpsPass
    : public impl::AMDAIEFlattenVectorizedOpsBase<
          AMDAIEFlattenVectorizedOpsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, memref::MemRefDialect>();
  }

  AMDAIEFlattenVectorizedOpsPass() = default;
  AMDAIEFlattenVectorizedOpsPass(const AMDAIEFlattenVectorizedOpsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFlattenVectorizedOpsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();
  IRRewriter rewriter(context);
  // TODO(avarma): Currently this is fixated on just `arith.truncf`. Follow-up
  //               on this later to generalize.
  moduleOp->walk([&](arith::TruncFOp op) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    // Get old shape type.
    auto oldShapedType = cast<ShapedType>(op.getType());
    // Linearize the shape.
    int64_t linearizedSize = oldShapedType.getNumElements();
    // Fetch input(s).
    Value origInputOfTruncFOp = op.getIn();
    // Form linearized vector shape type for input and output.
    VectorType newVectorTypeForInput = VectorType::get(
        {linearizedSize},
        cast<ShapedType>(origInputOfTruncFOp.getType()).getElementType());
    VectorType newVectorTypeForOutput =
        VectorType::get({linearizedSize}, oldShapedType.getElementType());
    // Shape cast the original input to linearized shape type.
    Value newInputVector = rewriter.create<vector::ShapeCastOp>(
        op.getLoc(), newVectorTypeForInput, origInputOfTruncFOp);
    // Create new base operation with the linearized input/output.
    Value newTruncFOp = rewriter.create<arith::TruncFOp>(
        op.getLoc(), newVectorTypeForOutput, newInputVector);
    // Delinearize the output back to the original type.
    Value newOutputVector = rewriter.create<vector::ShapeCastOp>(
        op.getLoc(), op.getType(), newTruncFOp);
    rewriter.replaceOp(op, newOutputVector);
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFlattenVectorizedOpsPass() {
  return std::make_unique<AMDAIEFlattenVectorizedOpsPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
