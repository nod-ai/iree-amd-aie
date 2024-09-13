// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-pack-to-dma"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Applies packing to a given input.
LogicalResult packDmaInputs(IREE::LinalgExt::PackOp packOp,
                            SmallVector<OpFoldResult> &offsets,
                            SmallVector<OpFoldResult> &sizes,
                            SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = packOp.getContext();

  llvm::ArrayRef<int64_t> permutation = packOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();

  SmallVector<OpFoldResult> innerSizes;
  SmallVector<OpFoldResult> innerStrides;
  SmallVector<OpFoldResult> innerOffsets;

  auto innerDimsPos = packOp.getInnerDimsPos();

  for (int i = 0; i < innerTiles.size(); i++) {
    // Calculate new sizes.
    innerSizes.push_back(getAsIndexOpFoldResult(ctx, innerTiles[i]));
    std::optional<int64_t> size = getConstantIntValue(sizes[innerDimsPos[i]]);
    assert(size.has_value() &&
           "expect constant index here in sizes vector of pack op");
    // Fail if tile doesnt perfectly divide the corresponding outer dim as we
    // do not support the padding semantics yet.
    if (size.value() % innerTiles[i] != 0) {
      auto message = llvm::formatv(
          "in dimension {0}, the tile size {1} does not divide the tensor size "
          "{2}. Imperfect/partial tiling is currently not supported.",
          i, innerTiles[i], size.value());
      return packOp->emitOpError(message);
    }

    sizes[innerDimsPos[i]] =
        getAsIndexOpFoldResult(ctx, size.value() / innerTiles[i]);
    // The tiled dim inherits the stride from the corresponding outer dim and
    // the outer dims stride gets multiplied by the size of the tile.
    innerStrides.push_back(strides[innerDimsPos[i]]);
    std::optional<int64_t> stride =
        getConstantIntValue(strides[innerDimsPos[i]]);
    assert(stride.has_value() &&
           "expect constant index in stride vector of pack op");
    strides[innerDimsPos[i]] =
        getAsIndexOpFoldResult(ctx, stride.value() * innerTiles[i]);
    // The tiled dim inherits the offset from the corresponding outer dim and
    // the outer dim offset is set to zero.
    innerOffsets.push_back(offsets[innerDimsPos[i]]);
    offsets[innerDimsPos[i]] = getAsIndexOpFoldResult(ctx, 0);
  }
  // Apply permutations to the outer dims if provided.
  if (!permutation.empty()) {
    applyPermutationToVector(strides, permutation);
    applyPermutationToVector(sizes, permutation);
    applyPermutationToVector(offsets, permutation);
  }
  // Merge the dims.
  sizes.insert(sizes.end(), innerSizes.begin(), innerSizes.end());
  strides.insert(strides.end(), innerStrides.begin(), innerStrides.end());
  offsets.insert(offsets.end(), innerOffsets.begin(), innerOffsets.end());
  return success();
}

/// Applies unpacking to a given input.
LogicalResult unPackDmaInputs(IREE::LinalgExt::UnPackOp unPackOp,
                              SmallVector<OpFoldResult> &offsets,
                              SmallVector<OpFoldResult> &sizes,
                              SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = unPackOp.getContext();

  llvm::ArrayRef<int64_t> permutation = unPackOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = unPackOp.getStaticInnerTiles();

  SmallVector<OpFoldResult> innerSizes;
  SmallVector<OpFoldResult> innerStrides;
  SmallVector<OpFoldResult> innerOffsets;
  auto innerDimsPos = unPackOp.getInnerDimsPos();

  int numOuterDims = sizes.size() - innerTiles.size();
  SmallVector<OpFoldResult> outerOffsets = SmallVector<OpFoldResult>(
      offsets.begin(), offsets.begin() + numOuterDims);
  SmallVector<OpFoldResult> outerStrides = SmallVector<OpFoldResult>(
      strides.begin(), strides.begin() + numOuterDims);
  SmallVector<OpFoldResult> outerSizes =
      SmallVector<OpFoldResult>(sizes.begin(), sizes.begin() + numOuterDims);

  // Apply permutations to the outer dims if provided.
  if (!permutation.empty()) {
    applyPermutationToVector(outerStrides, permutation);
    applyPermutationToVector(outerSizes, permutation);
    applyPermutationToVector(outerOffsets, permutation);
  }
  // Do the unpacking on the Outer dims.
  llvm::SmallDenseMap<int64_t, int64_t> outerDimsIndexMap;
  // Intialize the indexing of each outer dim.
  for (int i = 0; i < numOuterDims; i++) {
    outerDimsIndexMap[i] = i;
  }
  for (int i = 0; i < innerTiles.size(); i++) {
    // Insert inner dims adjacent to there corresponding outer dims.
    outerSizes.insert(
        outerSizes.begin() + outerDimsIndexMap[innerDimsPos[i]] + 1,
        getAsIndexOpFoldResult(ctx, innerTiles[i]));
    outerStrides.insert(
        outerStrides.begin() + outerDimsIndexMap[innerDimsPos[i]] + 1,
        strides[numOuterDims + i]);
    outerOffsets.insert(
        outerOffsets.begin() + outerDimsIndexMap[innerDimsPos[i]] + 1,
        offsets[numOuterDims + i]);
    // Update the map as all the dimensions inner to the innerDimsPos[i] are now
    // shifted by 1.
    for (int j = innerDimsPos[i] + 1; j < numOuterDims; j++) {
      outerDimsIndexMap[j]++;
    }
  }
  // Make the outer dims as the final returned dims
  offsets = outerOffsets;
  strides = outerStrides;
  sizes = outerSizes;
  return success();
}

/// Examines an input/output of a pack/unpack op and provides the
/// corresponding offsets, sizes and strides required by the dma op
LogicalResult setDmaInputs(Operation *&operandOp,
                           SmallVector<OpFoldResult> &offsets,
                           SmallVector<OpFoldResult> &sizes,
                           SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = operandOp->getContext();
  if (auto allocOp = dyn_cast<memref::AllocOp>(operandOp)) {
    auto [stridesI64, baseOffset] = getStridesAndOffset(allocOp.getType());
    if (baseOffset != 0) {
      auto message = llvm::formatv(
          "with non-zero base offset {0} is not supported by the "
          "current pass, requires testing and possible code changes.",
          baseOffset);
      return allocOp->emitOpError(message);
    }
    strides = getAsIndexOpFoldResult(ctx, stridesI64);
    auto sizesI64 = allocOp.getType().getShape();
    if (llvm::any_of(sizesI64, [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return allocOp->emitOpError(
          "with dynamic shape is not supported by dma op.");
    }
    sizes = getAsIndexOpFoldResult(ctx, sizesI64);
    // Alloc Op has no offsets.
    for (int i = 0; i < sizes.size(); i++) {
      offsets.push_back(getAsIndexOpFoldResult(ctx, 0));
    }
    return success();
  }
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(operandOp)) {
    auto mixedStrides = subviewOp.getMixedStrides();
    if (llvm::any_of(mixedStrides, [](OpFoldResult ofr) {
          return !isConstantIntValue(ofr, 1);
        })) {
      auto message = llvm::formatv(
          "has non-unit mixed strides that are not currently supported by this "
          "pass.");
      return subviewOp->emitOpError(message);
    }
    offsets = subviewOp.getMixedOffsets();
    auto [stridesI64, baseOffset] =
        getStridesAndOffset(subviewOp.getSource().getType());
    if (baseOffset != 0) {
      auto message = llvm::formatv(
          "has non-zero base offset {0} that is not supported by the "
          "current pass: requires testing and possible code changes.",
          baseOffset);
      return subviewOp->emitOpError(message);
    }
    strides = getAsIndexOpFoldResult(ctx, stridesI64);
    operandOp = subviewOp.getSource().getDefiningOp();
    auto sizesI64 = subviewOp.getType().getShape();
    if (llvm::any_of(sizesI64, [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return subviewOp->emitOpError(
          "has dynamic shape that is not supported by the target dma op.");
    }
    sizes = getAsIndexOpFoldResult(ctx, sizesI64);
    return success();
  }
  return operandOp->emitOpError(
      "is an unsupported operation. This pass currently only supports AllocOp "
      "and SubViewOp as inputs.");
}

/// Get the inputs from the pack/unpack op 'op'. Return failure if 'op' is not
/// a pack/unpack op, or if 'op' is determined unlowerable to a DMA operation.
LogicalResult processInputs(Operation *op, SmallVector<OpFoldResult> &offsets,
                            SmallVector<OpFoldResult> &sizes,
                            SmallVector<OpFoldResult> &strides) {
  if (auto packOp = dyn_cast<IREE::LinalgExt::PackOp>(op)) {
    if (failed(packDmaInputs(packOp, offsets, sizes, strides))) {
      return failure();
    }
  } else if (auto unPackOp = dyn_cast<IREE::LinalgExt::UnPackOp>(op)) {
    if (failed(unPackDmaInputs(unPackOp, offsets, sizes, strides))) {
      return failure();
    }
  } else {
    return failure();
  }
  return success();
}

/// Rewrite the pack/unpack op 'op' as a DMA operation. The function arguments
/// 'input', 'output', and 'innerTiles' are the input, output, and inner tile
/// of 'op'. If 'op' is not a pack/unpack op, or if it determined to not
/// currently be lowerable to a DMA operation, failure is returned.
///
/// Design note: arguments 'input', 'output', and 'innerTiles' could be
/// obtained from 'op' inside this function if it were templatized, but
/// I've factorized out that logic to reduce the total amount of templatized
/// code.
LogicalResult rewriteAsDma(IRRewriter &rewriter, Operation *op, Value input,
                           Value output, llvm::ArrayRef<int64_t> innerTiles) {
  if (llvm::any_of(innerTiles,
                   [](int64_t size) { return ShapedType::isDynamic(size); })) {
    op->emitError("has a non-static shape: not yet supported by this pass.");
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  Operation *sourceOp = input.getDefiningOp();
  Operation *dstOp = output.getDefiningOp();

  // Prepare source DMA inputs.
  SmallVector<OpFoldResult> srcOffsets;
  SmallVector<OpFoldResult> srcBaseStrides;
  SmallVector<OpFoldResult> srcShape;

  if (!succeeded(
          setDmaInputs(sourceOp, srcOffsets, srcShape, srcBaseStrides))) {
    return failure();
  }

  if (!succeeded(processInputs(op, srcOffsets, srcShape, srcBaseStrides))) {
    return failure();
  }

  // Prepare destination DMA inputs.
  SmallVector<OpFoldResult> dstOffsets;
  SmallVector<OpFoldResult> dstBaseStrides;
  SmallVector<OpFoldResult> dstShape;
  if (!succeeded(setDmaInputs(dstOp, dstOffsets, dstShape, dstBaseStrides))) {
    return failure();
  }

  // Create logical objectFifos from source and destination memrefs.
  Value srcVal = sourceOp->getResult(0);
  Value dstVal = dstOp->getResult(0);
  auto srcType = cast<MemRefType>(srcVal.getType());
  auto dstType = cast<MemRefType>(dstVal.getType());

  rewriter.setInsertionPointAfter(srcVal.getDefiningOp());
  auto src = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
      rewriter.getUnknownLoc(), LogicalObjectFifoType::get(srcType), srcVal);
  rewriter.setInsertionPointAfter(dstVal.getDefiningOp());
  auto dst = rewriter.create<AMDAIE::LogicalObjectFifoFromMemrefOp>(
      rewriter.getUnknownLoc(), LogicalObjectFifoType::get(dstType), dstVal);

  rewriter.setInsertionPoint(op);
  rewriter.create<AMDAIE::DmaCpyNdOp>(op->getLoc(), dst, dstOffsets, dstShape,
                                      dstBaseStrides, src, srcOffsets, srcShape,
                                      srcBaseStrides);
  rewriter.eraseOp(op);
  return success();
}

template <typename PackOrUnpackOp>
LogicalResult rewriteAsDma(PackOrUnpackOp op, IRRewriter &rewriter) {
  Value input = op.getInput();
  Value output = op.getOutput();
  llvm::ArrayRef<int64_t> innerTiles = op.getStaticInnerTiles();
  return rewriteAsDma(rewriter, op, input, output, innerTiles);
}

};  // namespace

class AMDAIEPackToDmaPass
    : public impl::AMDAIEPackToDmaBase<AMDAIEPackToDmaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    IREE::LinalgExt::IREELinalgExtDialect, AMDAIEDialect>();
  }

  AMDAIEPackToDmaPass() = default;
  AMDAIEPackToDmaPass(const AMDAIEPackToDmaPass &pass){};
  void runOnOperation() override;
};

void AMDAIEPackToDmaPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  auto walkResult = getOperation()->walk(
      [&rewriter](IREE::LinalgExt::PackOp op) -> WalkResult {
        if (failed(rewriteAsDma(op, rewriter))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted()) signalPassFailure();
  walkResult = getOperation()->walk(
      [&rewriter](IREE::LinalgExt::UnPackOp op) -> WalkResult {
        if (failed(rewriteAsDma(op, rewriter))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted()) signalPassFailure();
}

std::unique_ptr<Pass> createAMDAIEPackToDmaPass() {
  return std::make_unique<AMDAIEPackToDmaPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
