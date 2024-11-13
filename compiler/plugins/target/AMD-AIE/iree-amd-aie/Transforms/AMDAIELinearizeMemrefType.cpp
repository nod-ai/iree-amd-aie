// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-amdaie-linearize-memref-type"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIELinearizeMemrefTypePass
    : public impl::AMDAIELinearizeMemrefTypeBase<
          AMDAIELinearizeMemrefTypePass> {
 public:
  AMDAIELinearizeMemrefTypePass() = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override;
};

void AMDAIELinearizeMemrefTypePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  unsigned uniqueOutlinedMatmul = 0;
  unsigned uniqueOutlinedElementwise = 0;
  DenseMap<Operation *, std::string> computeOpToOutlinedFuncMap;
  SmallVector<Operation *> toBeErased;
  moduleOp.walk([&](Operation *op) {
    // TODO(avarma): Except funcOps. This will be improved later. And the
    // following will be pulled out to a PatterRewriter later.
    auto currentType = dyn_cast<MemRefType>(op.getType());
    if (!currentType) {
      return;
      // return rewriter.notifyMatchFailure(op->getLoc(),
      //                                    "unhandled non-memref types");
    }
    // Convert current type later.
    auto newResultType = dyn_cast<MemRefType>(convertCurrentType);
    if (!newResultType) {
      return;
      // return rewriter.notifyMatchFailure(
      //     op->getLoc(),
      //     llvm::formatv("failed to legalize memref type: {0}",
      //     op.getType()));
    }
    Location loc = op.getLoc();
    OpFoldResult zero = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> indices(currentType.getRank(), zero);

    // Get linearized type.
    int srcBits = currentType.getElementType().getIntOrFloatBitWidth();
    int dstBits = newResultType.getElementType().getIntOrFloatBitWidth();
    OpFoldResult elementOffset;
    Value byteOffset = adaptor.getByteOffset();
    if (byteOffset && !matchPattern(byteOffset, m_Zero())) {
      elementOffset = convertByteOffsetToElementOffset(
          rewriter, loc, byteOffset, currentType.getElementType());
    } else {
      elementOffset = rewriter.getIndexAttr(0);
    }
    SmallVector<OpFoldResult> sizes = getMixedValues(
        currentType.getShape(), adaptor.getDynamicDims(), rewriter);
    memref::LinearizedMemRefInfo linearizedMemRefInfo =
        memref::getLinearizedMemRefOffsetAndSize(rewriter, loc, srcBits,
                                                 dstBits, elementOffset, sizes);

    SmallVector<Value> dynamicLinearizedSize;
    if (newResultType.getRank() > 0 && !newResultType.hasStaticShape()) {
      dynamicLinearizedSize.push_back(getValueOrCreateConstantIndexOp(
          rewriter, loc, linearizedMemRefInfo.linearizedSize));
    }
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELinearizeMemrefTypePass() {
  return std::make_unique<AMDAIELinearizeMemrefTypePass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
