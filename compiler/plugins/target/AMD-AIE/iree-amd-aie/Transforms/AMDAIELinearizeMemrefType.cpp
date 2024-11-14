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

static SmallVector<int64_t> getLinearizedShape(MemRefType ty, int srcBits,
                                               int dstBits) {
  if (ty.getRank() == 0) return {};

  int64_t linearizedShape = 1;
  for (auto shape : ty.getShape()) {
    if (shape == ShapedType::kDynamic) return {ShapedType::kDynamic};
    linearizedShape *= shape;
  }
  int scale = dstBits / srcBits;
  // Scale the size to the ceilDiv(linearizedShape, scale)
  // to accomodate all the values.
  linearizedShape = (linearizedShape + scale - 1) / scale;
  return {linearizedShape};
}

static LogicalResult linearizeType(MemRefType memrefType,
                                   MemRefType &newMemrefType) {
  // Fetch linearized shape.
  // TODO(avarma): Take into account different src/dst bits.
  int srcBits = memrefType.getElementType().getIntOrFloatBitWidth();
  SmallVector<int64_t> linearizedShape =
      getLinearizedShape(memrefType, srcBits, srcBits);
  // Fetch offset and strides of the old memref.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memrefType, strides, offset)))
    return failure();
  if (!strides.empty() && strides.back() != 1) return failure();
  // Form layout for the linearized memref.
  StridedLayoutAttr layoutAttr;
  // If the offset is 0, we do not need a strided layout as the stride is
  // 1, so we only use the strided layout if the offset is not 0.
  if (offset != 0) {
    if (offset == ShapedType::kDynamic) {
      layoutAttr = StridedLayoutAttr::get(memrefType.getContext(), offset,
                                          ArrayRef<int64_t>{1});
    } else {
      // TODO(avarma): Take into account different src/dst bits.
      // // Check if the number of bytes are a multiple of the loadStoreWidth
      // // and if so, divide it by the loadStoreWidth to get the offset.
      // if ((offset * width) % loadStoreWidth != 0)
      //   return std::nullopt;
      // offset = (offset * width) / loadStoreWidth;

      layoutAttr = StridedLayoutAttr::get(memrefType.getContext(), offset,
                                          ArrayRef<int64_t>{1});
    }
  }
  Type elementType = memrefType.getElementType();
  newMemrefType = MemRefType::get(linearizedShape, elementType, layoutAttr,
                                  memrefType.getMemorySpace());
  return success();
}

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

  moduleOp.walk([&](Operation *op) {
    if (isa<func::FuncOp, func::ReturnOp, arith::ConstantOp>(op))
      return WalkResult::skip();
    // TODO(avarma): Except funcOps. This will be improved later. And the
    // following will be pulled out to a PatterRewriter later.
    for (Value operand : op->getOperands()) {
      auto currentType = dyn_cast<MemRefType>(operand.getType());
      if (!currentType) {
        continue;
        // return rewriter.notifyMatchFailure(op->getLoc(),
        //                                    "unhandled non-memref types");
      }
      // Convert current type later.
      MemRefType newResultType;
      if (failed(linearizeType(currentType, newResultType)))
        return WalkResult::interrupt();

      // if (!newResultType) {
      //     return;
      //     // return rewriter.notifyMatchFailure(
      //     //     op->getLoc(),
      //     //     llvm::formatv("failed to legalize memref type: {0}",
      //     //     op.getType()));
      //   }
      // Location loc = op->getLoc();
      OpFoldResult zero = rewriter.getIndexAttr(0);
      SmallVector<OpFoldResult> indices(currentType.getRank(), zero);

      // Get linearized type.
      // int srcBits = currentType.getElementType().getIntOrFloatBitWidth();
      // int dstBits = newResultType.getElementType().getIntOrFloatBitWidth();
      llvm::outs() << "SRC TYPE := " << currentType << "\n";
      llvm::outs() << "NEW TYPE := " << newResultType << "\n";
      // OpFoldResult elementOffset;
      // Value byteOffset = adaptor.getByteOffset();
      // if (byteOffset && !matchPattern(byteOffset, m_Zero())) {
      //   elementOffset = convertByteOffsetToElementOffset(
      //       rewriter, loc, byteOffset, currentType.getElementType());
      // } else {
      //   elementOffset = rewriter.getIndexAttr(0);
      // }

      // llvm::outs()<<"AFFINE MAP :=
      // "<<currentType.getLayout().getAffineMap()<<"\n"; llvm::outs().flush();
      // SmallVector<OpFoldResult> sizes = getMixedValues(
      //     currentType.getShape(), adaptor.getDynamicDims(), rewriter);
      // memref::LinearizedMemRefInfo linearizedMemRefInfo =
      //     memref::getLinearizedMemRefOffsetAndSize(rewriter, loc, srcBits,
      //                                              dstBits, elementOffset,
      //                                              sizes);

      //   SmallVector<Value> dynamicLinearizedSize;
      //   if (newResultType.getRank() > 0 && !newResultType.hasStaticShape()) {
      //     dynamicLinearizedSize.push_back(getValueOrCreateConstantIndexOp(
      //         rewriter, loc, linearizedMemRefInfo.linearizedSize));
      //   }
    }
    return WalkResult::advance();
  });
  return;
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELinearizeMemrefTypePass() {
  return std::make_unique<AMDAIELinearizeMemrefTypePass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
