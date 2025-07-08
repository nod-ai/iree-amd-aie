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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-convert-to-dma"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Applies dma transposition on the side that has lower number of dimensions,
/// which means the source side for pack ops and the destination side for unpack
/// ops.
template <typename PackOrUnpackOp>
LogicalResult dmaTransposeOnLowerNumDims(PackOrUnpackOp packOrUnpackOp,
                                         SmallVector<OpFoldResult> &offsets,
                                         SmallVector<OpFoldResult> &sizes,
                                         SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = packOrUnpackOp.getContext();

  llvm::ArrayRef<int64_t> permutation = packOrUnpackOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = packOrUnpackOp.getStaticInnerTiles();

  SmallVector<OpFoldResult> innerSizes;
  SmallVector<OpFoldResult> innerStrides;
  SmallVector<OpFoldResult> innerOffsets;

  ArrayRef<int64_t> innerDimsPos = packOrUnpackOp.getInnerDimsPos();

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
      return packOrUnpackOp->emitOpError(message);
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

/// Applies dma transposition on the side which has higher number of dimensions,
/// which means the destination side for pack ops and the source side for unpack
/// ops.
template <typename PackOrUnpackOp>
LogicalResult dmaTransposeOnHigherNumDims(PackOrUnpackOp packOrUnpackOp,
                                          SmallVector<OpFoldResult> &offsets,
                                          SmallVector<OpFoldResult> &sizes,
                                          SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = packOrUnpackOp.getContext();

  llvm::ArrayRef<int64_t> permutation = packOrUnpackOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = packOrUnpackOp.getStaticInnerTiles();

  SmallVector<OpFoldResult> innerSizes;
  SmallVector<OpFoldResult> innerStrides;
  SmallVector<OpFoldResult> innerOffsets;
  ArrayRef<int64_t> innerDimsPos = packOrUnpackOp.getInnerDimsPos();

  int numOuterDims = sizes.size() - innerTiles.size();
  SmallVector<OpFoldResult> outerOffsets = SmallVector<OpFoldResult>(
      offsets.begin(), offsets.begin() + numOuterDims);
  SmallVector<OpFoldResult> outerStrides = SmallVector<OpFoldResult>(
      strides.begin(), strides.begin() + numOuterDims);
  SmallVector<OpFoldResult> outerSizes =
      SmallVector<OpFoldResult>(sizes.begin(), sizes.begin() + numOuterDims);

  // Apply inverse permutation to the outer dims if permutation provided (if
  // permutation not provided, it is identity, and therefore so is the inverse).
  if (!permutation.empty()) {
    SmallVector<int64_t> inversePermutation =
        invertPermutationVector(permutation);
    applyPermutationToVector(outerStrides, inversePermutation);
    applyPermutationToVector(outerSizes, inversePermutation);
    applyPermutationToVector(outerOffsets, inversePermutation);
  }

  // Initialize the indexing of each outer dim.
  llvm::SmallDenseMap<int64_t, int64_t> outerDimsIndexMap;
  for (int i = 0; i < numOuterDims; i++) {
    outerDimsIndexMap[i] = i;
  }

  // Update outer dim sizes/strides/offsts.
  for (int i = 0; i < innerTiles.size(); i++) {
    // Insert inner dims adjacent to their corresponding outer dims.
    int insertionIndex = outerDimsIndexMap[innerDimsPos[i]] + 1;
    outerSizes.insert(outerSizes.begin() + insertionIndex,
                      getAsIndexOpFoldResult(ctx, innerTiles[i]));
    outerStrides.insert(outerStrides.begin() + insertionIndex,
                        strides[numOuterDims + i]);
    outerOffsets.insert(outerOffsets.begin() + insertionIndex,
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
/// corresponding offsets, sizes and strides required by the dma op.
LogicalResult setDmaInputs(Operation *&operandOp,
                           SmallVector<OpFoldResult> &offsets,
                           SmallVector<OpFoldResult> &sizes,
                           SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = operandOp->getContext();
  if (isa<memref::AllocOp>(operandOp) ||
      isa<memref::AssumeAlignmentOp>(operandOp)) {
    MemRefType memRefType = cast<MemRefType>(operandOp->getResult(0).getType());
    auto [stridesI64, baseOffset] = memRefType.getStridesAndOffset();
    if (baseOffset != 0) {
      auto message = llvm::formatv(
          "with non-zero base offset {0} is not supported by the "
          "current pass, requires testing and possible code changes.",
          baseOffset);
      return operandOp->emitOpError(message);
    }
    strides = getAsIndexOpFoldResult(ctx, stridesI64);
    auto sizesI64 = memRefType.getShape();
    if (llvm::any_of(sizesI64, [](int64_t size) {
          return ShapedType::isDynamic(size);
        })) {
      return operandOp->emitOpError(
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
    MemRefType subviewType = subviewOp.getSource().getType();
    auto [stridesI64, baseOffset] = subviewType.getStridesAndOffset();
    if (baseOffset != 0) {
      auto message = llvm::formatv(
          "has non-zero base offset {0} that is not supported by the "
          "current pass: requires testing and possible code changes.",
          baseOffset);
      return subviewOp->emitOpError(message);
    }
    strides = getAsIndexOpFoldResult(ctx, stridesI64);
    operandOp = subviewOp.getSource().getDefiningOp();
    sizes = subviewOp.getMixedSizes();
    if (llvm::any_of(sizes, [](OpFoldResult fr) {
          return !getConstantIntValue(fr).has_value();
        })) {
      return subviewOp->emitOpError(
          " has dynamic shape that is not supported by the target dma op.");
    }

    assert(offsets.size() == sizes.size() && sizes.size() == strides.size() &&
           "mismatch in the number of offsets, sizes and strides");

    // Handle the case where some dimensions are dropped in the subview:
    llvm::SmallBitVector droppedDims = subviewOp.getDroppedDims();
    uint64_t insertionIndex{0};
    for (uint64_t extractionIndex = 0; extractionIndex < offsets.size();
         ++extractionIndex) {
      if (!droppedDims[extractionIndex]) {
        offsets[insertionIndex] = offsets[extractionIndex];
        sizes[insertionIndex] = sizes[extractionIndex];
        strides[insertionIndex] = strides[extractionIndex];
        insertionIndex++;
      }
    }
    offsets.resize(insertionIndex);
    sizes.resize(insertionIndex);
    strides.resize(insertionIndex);
    return success();
  }
  return operandOp->emitOpError(
      "is an unsupported operation. This pass currently only supports "
      "memref.assume_alignment, memref.alloc and memref.subview as "
      "inputs.");
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
template <typename PackOrUnpackOp>
LogicalResult rewriteAsDma(IRRewriter &rewriter, PackOrUnpackOp op, Value input,
                           Value output, llvm::ArrayRef<int64_t> innerTiles,
                           bool transposeOnSource) {
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
  SmallVector<OpFoldResult> srcStrides;
  SmallVector<OpFoldResult> srcShape;
  if (failed(setDmaInputs(sourceOp, srcOffsets, srcShape, srcStrides))) {
    return failure();
  }

  // Prepare destination DMA inputs.
  SmallVector<OpFoldResult> dstOffsets;
  SmallVector<OpFoldResult> dstStrides;
  SmallVector<OpFoldResult> dstShape;
  if (failed(setDmaInputs(dstOp, dstOffsets, dstShape, dstStrides))) {
    return failure();
  }

  // Update dma source or destination addressing based on the side for dma
  // transposition.
  {
    SmallVector<OpFoldResult> &offsets =
        transposeOnSource ? srcOffsets : dstOffsets;

    SmallVector<OpFoldResult> &shape = transposeOnSource ? srcShape : dstShape;

    SmallVector<OpFoldResult> &strides =
        transposeOnSource ? srcStrides : dstStrides;

    bool sourceIsHigherDim = dstStrides.size() <= srcStrides.size();

    if (sourceIsHigherDim == transposeOnSource) {
      if (failed(dmaTransposeOnHigherNumDims(op, offsets, shape, strides))) {
        return failure();
      }
    } else {
      if (failed(dmaTransposeOnLowerNumDims(op, offsets, shape, strides))) {
        return failure();
      }
    }
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
                                      dstStrides, src, srcOffsets, srcShape,
                                      srcStrides);
  rewriter.eraseOp(op);
  return success();
}

template <typename PackOrUnpackOp>
LogicalResult rewriteAsDma(PackOrUnpackOp op, IRRewriter &rewriter,
                           bool tranposeOnSource) {
  Value input = op.getInput();
  Value output = op.getOutput();
  llvm::ArrayRef<int64_t> innerTiles = op.getStaticInnerTiles();
  return rewriteAsDma(rewriter, op, input, output, innerTiles,
                      tranposeOnSource);
}

/// Convert a linalg.copy operation on 2 memrefs to an equivalent pack/unpack
/// operation. If the linalg.copy operation is to a memory closer to the
/// core it is converted to a pack operation, otherwise an unpack operation.
///
/// Note: we could convert all copies to packs, but it would be potentially
/// confusing to have packs ops moving data away from cores.
LogicalResult copyToPack(IRRewriter &rewriter, linalg::CopyOp copyOp) {
  if (copyOp.getNumOperands() != 2 || copyOp.getNumResults() != 0) {
    copyOp.emitOpError()
        << "has " << copyOp.getNumOperands() << " operands and "
        << copyOp.getNumResults()
        << " results. It must have 2 operands and 0 results to convert "
           "to an iree.linalg_ext dialect pack/unpack operation";
    return failure();
  }
  // Setting up the 'identity' pack/unpack:
  ArrayRef<int64_t> innerDimsPos{};
  ArrayRef<OpFoldResult> innerTiles{};

  Value src = copyOp.getOperand(0);
  Value dst = copyOp.getOperand(1);

  // MemRefTypes with no memory space attribute return 0 here, so this is safe.
  uint32_t srcMemspace = cast<MemRefType>(src.getType()).getMemorySpaceAsInt();
  uint32_t dstMemspace = cast<MemRefType>(dst.getType()).getMemorySpaceAsInt();
  const bool towardsCore = srcMemspace <= dstMemspace;

  rewriter.setInsertionPoint(copyOp);
  if (towardsCore) {
    rewriter.replaceOpWithNewOp<IREE::LinalgExt::PackOp>(
        copyOp, src, dst, innerDimsPos, innerTiles);
  } else {
    rewriter.replaceOpWithNewOp<IREE::LinalgExt::UnPackOp>(
        copyOp, src, dst, innerDimsPos, innerTiles);
  }

  return success();
}

};  // namespace

class AMDAIEConvertToDmaPass
    : public impl::AMDAIEConvertToDmaBase<AMDAIEConvertToDmaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    IREE::LinalgExt::IREELinalgExtDialect, AMDAIEDialect>();
  }

  AMDAIEConvertToDmaPass() = default;
  AMDAIEConvertToDmaPass(const AMDAIEConvertToDmaPass &pass){};
  AMDAIEConvertToDmaPass(const AMDAIEConvertToDmaOptions &options)
      : AMDAIEConvertToDmaBase(options) {}

  void runOnOperation() override;
};

void AMDAIEConvertToDmaPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  // Convert all linalg.copy to iree_linalg_ext.pack/unpack ops. We then
  // bootstrap the work done for lowering the pack/unpack op to dmas as the next
  // step. This is easy to implement, but not the most direct lowering, so
  // we might want to revisit this.
  WalkResult convertCopiesWalkResult =
      getOperation()->walk([&](linalg::CopyOp copyOp) {
        if (failed(copyToPack(rewriter, copyOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  if (convertCopiesWalkResult.wasInterrupted()) return signalPassFailure();

  WalkResult walkResult =
      getOperation()->walk([&](IREE::LinalgExt::PackOp packOp) {
        if (failed(rewriteAsDma(packOp, rewriter, packTransposeOnSource))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted()) signalPassFailure();

  walkResult = getOperation()->walk([&](IREE::LinalgExt::UnPackOp unpackOp) {
    if (failed(rewriteAsDma(unpackOp, rewriter, unpackTransposeOnSource))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) signalPassFailure();
}

std::unique_ptr<Pass> createAMDAIEConvertToDmaPass(
    AMDAIEConvertToDmaOptions options) {
  return std::make_unique<AMDAIEConvertToDmaPass>(options);
}
}  // namespace mlir::iree_compiler::AMDAIE
