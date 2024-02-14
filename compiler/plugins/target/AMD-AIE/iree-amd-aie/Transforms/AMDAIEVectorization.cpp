// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-vectorization"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEVectorizationPass
    : public impl::AMDAIEVectorizationBase<AMDAIEVectorizationPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<scf::SCFDialect>();
  }

  AMDAIEVectorizationPass() = default;
  AMDAIEVectorizationPass(const AMDAIEVectorizationPass &) = default;

  // We only vectorize linalg.matmuls of a specific shape. The shape is chosen
  // to target AIE instructions, and is dependent on the data type.
  // Matmuls of the form A: 4XK, B: KX4, C: 4X4 are vectorized, where K is type
  // specific. This function returns K based on data type. 
  static FailureOr<uint32_t> getMatmulTargetK(mlir::linalg::MatmulOp op) {
    auto elType =
        op->getResult(0).getType().cast<ShapedType>().getElementType();
    if (elType.isF16() || elType.isBF16() || elType.isInteger(16))
      return 8;
    else if (elType.isF32() || elType.isInteger(32))
      return 4;
    return op->emitOpError("Unimplemented: reduction size for type") << elType;
  }

  // Return true if 'op' is a matmul op of the correct shape. 
  static FailureOr<bool> isMatmulCandidate(Operation *op) {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(op);
    if (!matmulOp){
      return false;
    }
    auto lhsType = op->getOperand(0).getType().cast<ShapedType>();
    auto rhsType = op->getOperand(1).getType().cast<ShapedType>();
    auto outType = op->getResult(0).getType().cast<ShapedType>();

    // TODO: matmuls with mixed types? 
    auto elType = outType.getElementType();
    if (lhsType.getElementType() != elType ||
        rhsType.getElementType() != elType) {
      return false;
    }

    auto rhsShape = rhsType.getShape();
    auto outShape = outType.getShape();

    assert(outShape.size() == 2 && rhsShape.size() == 2 &&
           "linalg.matmuls must be rank-2");

    auto targetK = getMatmulTargetK(matmulOp);
    if (failed(targetK)) return failure();

    auto m = outShape[0];
    if (m != 4) return false;
    auto n = rhsShape[1];
    if (n != 4) return false;
    auto k = rhsShape[0];
    if (k != targetK) return false;

    return true;
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    auto funcOp = getOperation();
    if (!llvm::isa<mlir::func::FuncOp>(funcOp)) {
      funcOp->emitError("expected a function operation");
      return signalPassFailure();
    }

    // Collect all the operations which should be vectorized.
    SmallVector<Operation *> opsToVectorize;
    funcOp.walk([&](Operation *op) {
      auto candidate = isMatmulCandidate(op);
      if (failed(candidate)) return signalPassFailure();
      if (candidate.value()) opsToVectorize.push_back(op);
    });

    for (auto opToVectorize : opsToVectorize)
      if (failed(linalg::vectorize(rewriter, opToVectorize))) {
        opToVectorize->emitOpError("Unexpectedly failed to vectorize");
        return signalPassFailure();
      }
  }
};
}  // namespace

std::unique_ptr<Pass> createAMDAIEVectorizationPass() {
  return std::make_unique<AMDAIEVectorizationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
