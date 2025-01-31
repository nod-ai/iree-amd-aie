// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIEDmaUtils.h"

#include "AMDAIEUtils.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "iree-amdaie-dma-utils"

namespace mlir::iree_compiler::AMDAIE {

// Functions in namespace `detail` are not intended to be used outside of this
// file, but are exposed in the .h file for testing purposes.
namespace detail {

void matchStridesOfUnitDims(MLIRContext *ctx, ArrayRef<OpFoldResult> sizesX,
                            SmallVector<OpFoldResult> &stridesX,
                            SmallVector<OpFoldResult> &offsetsX,
                            ArrayRef<OpFoldResult> stridesY) {
  for (int i = 0; i < sizesX.size(); ++i) {
    if (stridesX[i] != stridesY[i]) {
      std::optional<int64_t> maybeIntSize = getConstantIntValue(sizesX[i]);
      if (maybeIntSize.has_value() && maybeIntSize.value() == 1) {
        std::optional<int64_t> maybeOffset = getConstantIntValue(offsetsX[i]);
        std::optional<int64_t> maybeStrideX = getConstantIntValue(stridesX[i]);
        std::optional<int64_t> maybeStrideY = getConstantIntValue(stridesY[i]);
        if (maybeOffset.has_value() && maybeStrideX.has_value() &&
            maybeStrideY.has_value()) {
          int64_t offset = maybeOffset.value();
          int64_t strideX = maybeStrideX.value();
          int64_t strideY = maybeStrideY.value();
          int64_t numerator = offset * strideX;
          int64_t denominator = strideY;
          if (numerator % denominator == 0) {
            offsetsX[i] = getAsIndexOpFoldResult(ctx, numerator / denominator);
            stridesX[i] = stridesY[i];
          }
        }
      }
    }
  }
}

std::optional<int64_t> getGlobalOffsetDifference(
    ArrayRef<OpFoldResult> offsetsX, ArrayRef<OpFoldResult> stridesX,
    ArrayRef<OpFoldResult> offsetsY, ArrayRef<OpFoldResult> stridesY) {
  assert(offsetsX.size() == stridesX.size() &&
         "expected same number of offsets and strides for X");
  assert(offsetsY.size() == stridesY.size() &&
         "expected same number of offsets and strides for Y");
  assert(offsetsX.size() == offsetsY.size() &&
         "expected same number of offsets for X and Y");

  // In this function we're computing the constant globalOffsetDifference:
  //
  //    sum_{d} offsetsA[d] * stridesA[d]  -
  //    sum_{d} offsetsB[d] * stridesB[d] .
  //
  // If all values in offsetsA, offsetsB, stridesA, stridesB are constant,
  // this is straightforward. If not, we need all the non-constant terms to
  // cancel. In the maps below, we store the terms with non-constants, and then
  // check at the end of this function that they've all cancelled. In
  // `valToConst` we store terms where one of offset and stride is constant, and
  // the other is not. In `valPairs`, we keep track of all the terms where
  // neither the stride nor the offset is constant.
  DenseMap<Value, int64_t> valToConst;
  auto incrementValToConst = [&](Value v, int64_t signedStride) {
    auto iter = valToConst.find(v);
    if (iter == valToConst.end()) {
      valToConst[v] = signedStride;
    } else {
      iter->second += signedStride;
    }
  };

  DenseMap<std::pair<Value, Value>, int64_t> valPairs;
  auto incrementValPairs = [&](Value v0, Value v1, int64_t sign) {
    std::pair<Value, Value> p0(v0, v1);
    auto iter0 = valPairs.find(p0);
    if (iter0 != valPairs.end()) {
      iter0->second += sign;
      return;
    }
    std::pair<Value, Value> p1(v1, v0);
    auto iter1 = valPairs.find(p1);
    if (iter1 != valPairs.end()) {
      iter1->second += sign;
      return;
    }
    valPairs.insert({p0, sign});
  };

  int64_t globalOffsetDifference{0};

  // Add the term `offset * stride * sign` to the global offset different,
  // triaging the different combinations of constant/non-constant.
  auto updateGlobalOffsetDifference = [&](OpFoldResult offset,
                                          OpFoldResult stride, int64_t sign) {
    std::optional<int64_t> cOffset = getConstantIntValue(offset);
    std::optional<int64_t> cStride = getConstantIntValue(stride);
    Value vOffset = dyn_cast<Value>(offset);
    Value vStride = dyn_cast<Value>(stride);

    if (!cOffset.has_value() && !cStride.has_value()) {
      incrementValPairs(vOffset, vStride, sign);
    } else if (cOffset.has_value() && cStride.has_value()) {
      globalOffsetDifference += sign * cOffset.value() * cStride.value();
    } else if (cOffset.has_value()) {
      incrementValToConst(cast<Value>(stride), sign * cOffset.value());
    } else if (cStride.has_value()) {
      incrementValToConst(cast<Value>(offset), sign * cStride.value());
    }
  };

  for (uint32_t i = 0; i < offsetsX.size(); ++i) {
    updateGlobalOffsetDifference(offsetsX[i], stridesX[i], 1);
    updateGlobalOffsetDifference(offsetsY[i], stridesY[i], -1);
  }

  // The cases where the non-constant terms did not all cancel, and so the
  // global offset difference could not be determined to be constant.
  if (llvm::any_of(valToConst, [](auto x) { return x.second != 0; })) {
    return std::nullopt;
  }
  if (llvm::any_of(valPairs, [](auto x) { return x.second != 0; })) {
    return std::nullopt;
  }

  return globalOffsetDifference;
}
}  // namespace detail

namespace {

/// Consider 2 access patterns X and Y, where the access pattern for Y has one
/// more dimension than the access pattern for X. This function inserts a
/// singleton dimension into the access pattern for X, at the first dimension
/// from the back where the access patterns differ.
///
/// For example if X and Y have access patterns
///
/// X:  (offset: [0, 0]    sizes: [2, 8]    strides: [8, 1])
/// Y:  (offset: [0, 0, 0] sizes: [2, 4, 8] strides: [8, 16, 1])
///
/// then X is transformed into
///
/// X:  (offset: [0, 0, 0] sizes: [2, 1, 8] strides: [8, 16, 1])
void insertUnitDimension(MLIRContext *ctx, SmallVector<OpFoldResult> &offsetsX,
                         SmallVector<OpFoldResult> &sizesX,
                         SmallVector<OpFoldResult> &stridesX,
                         ArrayRef<OpFoldResult> stridesY) {
  assert(stridesY.size() == stridesX.size() + 1 &&
         "expected Y's rank to be 1 greater than X's");
  uint32_t index = stridesX.size();
  while (index > 0) {
    if (stridesY[index] != stridesX[index - 1]) break;
    index--;
  }
  OpFoldResult zeroFoldResult = getAsIndexOpFoldResult(ctx, 0);
  OpFoldResult oneFoldResult = getAsIndexOpFoldResult(ctx, 1);
  sizesX.insert(sizesX.begin() + index, oneFoldResult);
  stridesX.insert(stridesX.begin() + index, stridesY[index]);
  offsetsX.insert(offsetsX.begin() + index, zeroFoldResult);
}

/// If access pattern `A` followed by `B` can be merged into a single access
/// pattern, merge `B` into `A` and return true. Otherwise return false.
bool mergeInFirst(MLIRContext *ctx, SmallVector<OpFoldResult> &offsetsA,
                  SmallVector<OpFoldResult> &sizesA,
                  SmallVector<OpFoldResult> &stridesA,
                  SmallVector<OpFoldResult> offsetsB,
                  SmallVector<OpFoldResult> sizesB,
                  SmallVector<OpFoldResult> stridesB) {
  // Two rank-0 patterns merge into a single rank-0 pattern.
  if (offsetsA.size() == 0 && offsetsB.size() == 0) return true;

  // Local canonicalization to improve opportunities for merging:
  if (sizesA.size() + 1 == sizesB.size()) {
    insertUnitDimension(ctx, offsetsA, sizesA, stridesA, stridesB);
  } else if (sizesB.size() + 1 == sizesA.size()) {
    insertUnitDimension(ctx, offsetsB, sizesB, stridesB, stridesA);
  } else if (sizesA.size() != sizesB.size()) {
    // If the ranks of the accesses differ by more than 1, it is impossible
    // to merge them (unless the higher ranked access pattern has 2+ leading
    // dimensions of size 1, which is being ignored for now).
    return false;
  }
  detail::matchStridesOfUnitDims(ctx, sizesA, stridesA, offsetsA, stridesB);
  detail::matchStridesOfUnitDims(ctx, sizesB, stridesB, offsetsB, stridesA);

  // Check that strides and sizes are compatible for merging.
  if (stridesA != stridesB) return false;
  if (ArrayRef<OpFoldResult>(sizesA).drop_front() !=
      ArrayRef<OpFoldResult>(sizesB).drop_front()) {
    return false;
  }

  std::optional<int64_t> maybeOffsetDifference =
      detail::getGlobalOffsetDifference(offsetsB, stridesB, offsetsA, stridesA);

  // The case where the global offset difference is not constant is difficult to
  // handle, unless we can prove that it is non-negative. Leaving this edge case
  // for future work.
  if (!maybeOffsetDifference.has_value()) return false;
  int64_t offsetDifference = maybeOffsetDifference.value();

  // The special case where the global offset difference exactly matches the
  // pattern of A, in this case no new dimension is needed when merging the
  // patterns.
  std::optional<int64_t> cStrideA0 = getConstantIntValue(stridesA[0]);
  std::optional<int64_t> cSizeA0 = getConstantIntValue(sizesA[0]);
  std::optional<int64_t> cSizeB0 = getConstantIntValue(sizesB[0]);
  if (cStrideA0 && cSizeA0 && cSizeB0) {
    int64_t extended = cSizeA0.value() * cStrideA0.value();
    int64_t sum = cSizeA0.value() + cSizeB0.value();
    if (offsetDifference == extended) {
      sizesA[0] = getAsIndexOpFoldResult(ctx, sum);
      return true;
    }
  }

  // This is the case where the 2 patterns don't connect seamlessly, and we need
  // to introduce a new dimension to contain a new offset.
  if (sizesA[0] == sizesB[0] && (offsetDifference >= 0)) {
    int32_t index = 0;
    std::optional<int64_t> cSizeA0 = getConstantIntValue(sizesA[0]);
    if (cSizeA0.has_value() && cSizeA0.value() == 1) ++index;
    OpFoldResult of = getAsIndexOpFoldResult(ctx, offsetDifference);
    sizesA.insert(sizesA.begin() + index, getAsIndexOpFoldResult(ctx, 2));
    stridesA.insert(stridesA.begin() + index, of);
    offsetsA.insert(offsetsA.begin() + index, getAsIndexOpFoldResult(ctx, 0));
    return true;
  }

  return false;
}

}  // namespace

LogicalResult combineAccessPatterns(
    MLIRContext *ctx, ArrayRef<OpFoldResult> offsetsA,
    ArrayRef<OpFoldResult> sizesA, ArrayRef<OpFoldResult> stridesA,
    ArrayRef<OpFoldResult> offsetsB, ArrayRef<OpFoldResult> sizesB,
    ArrayRef<OpFoldResult> stridesB, SmallVector<OpFoldResult> &newOffsets,
    SmallVector<OpFoldResult> &newSizes, SmallVector<OpFoldResult> &newStrides,
    function_ref<bool(size_t)> exceedsNbDims) {
  assert(offsetsA.size() == sizesA.size() &&
         "expected same number of source offsets and sizes");
  assert(offsetsA.size() == stridesA.size() &&
         "expected same number of source offsets and strides");
  assert(offsetsB.size() == sizesB.size() &&
         "expected same number of source offsets and sizes");
  assert(offsetsB.size() == stridesB.size() &&
         "expected same number of source offsets and strides");

  // Ensure that OpFoldResults are Attributes when they can be. Specifally
  // this will replace arith.constant values with attributes.
  auto simplified =
      [&](ArrayRef<OpFoldResult> input) -> SmallVector<OpFoldResult> {
    SmallVector<OpFoldResult> x(input.begin(), input.end());
    for (OpFoldResult &y : x) {
      std::optional<int64_t> c = getConstantIntValue(y);
      if (c.has_value()) y = getAsIndexOpFoldResult(ctx, c.value());
    }
    return x;
  };

  newOffsets = simplified(offsetsA);
  newSizes = simplified(sizesA);
  newStrides = simplified(stridesA);
  SmallVector<OpFoldResult> mutableOffsetsB = simplified(offsetsB);
  SmallVector<OpFoldResult> mutableSizesB = simplified(sizesB);
  SmallVector<OpFoldResult> mutableStridesB = simplified(stridesB);

  bool combined = mergeInFirst(ctx, newOffsets, newSizes, newStrides,
                               mutableOffsetsB, mutableSizesB, mutableStridesB);

  // This is the case where the patterns could not be combined, even before the
  // check for exceeding the number of dimensions.
  if (!combined) return failure();
  (void)foldUnitDims(ctx, newOffsets, newSizes, newStrides);
  if (exceedsNbDims(newOffsets.size())) return failure();

  return success();
}

/// Fold subsequent dimensions within a strided access pattern that describe a
/// single linear access.
///
/// Example:
///
/// `offsets: [0, 0], sizes: [2, 8], strides: [8, 1]`
///
/// This describes accessing two times eight elements with a stride of eight in
/// between. For example if the input is: [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
/// 2, 2, 2, 2] this will access [1, 1, 1, 1, 1, 1, 1, 1] and then [2, 2, 2, 2,
/// 2, 2, 2, 2].
///
/// This is the same access pattern as accessing
/// [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2] in one go, which is
/// represented by:
///
/// `offsets: [0], sizes: [16], strides: [1]`
LogicalResult foldLinearDims(
    MLIRContext *ctx, const SmallVector<OpFoldResult> &offsets,
    const SmallVector<OpFoldResult> &sizes,
    const SmallVector<OpFoldResult> &strides,
    SmallVector<OpFoldResult> &newOffsets, SmallVector<OpFoldResult> &newSizes,
    SmallVector<OpFoldResult> &newStrides,
    function_ref<bool(size_t, int64_t)> checkValidSize) {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size() &&
         "expected same number of offsets, sizes and strides");
  bool foldableLinearDimsFound = false;
  if (offsets.size() == 0) return success(foldableLinearDimsFound);

  std::optional<SmallVector<int64_t>> staticSizes = getConstantIntValues(sizes);
  std::optional<SmallVector<int64_t>> staticStrides =
      getConstantIntValues(strides);
  if (!staticSizes || !staticStrides) return failure();
  SmallVector<int64_t> staticSizeVals = staticSizes.value();
  SmallVector<int64_t> staticStrideVals = staticStrides.value();

  newOffsets.push_back(offsets[offsets.size() - 1]);
  newStrides.push_back(strides[strides.size() - 1]);
  newSizes.push_back(sizes[sizes.size() - 1]);

  for (int i = offsets.size() - 2; i >= 0; i--) {
    // Conditions for folding a dim.
    // 1. Either, offsets[i] == 0 and then we can fold with any `newOffsets[-1]`
    // (even dynamic ones), OR offsets[i] multiplied by the respective stride,
    // is a multiple of the previous stride.
    // 2. newSizes[-1] x newStrides[-1] == strides[i]. With this we can have
    // newSizes[-1] = sizes[i] * newSizes[-1] , and then fold away the i
    // dimension
    // 3. checkValidSize(sizes[i] * newSizes[-1]). This allows hardware
    // constraints to be checked.
    size_t vecSize = newOffsets.size();
    std::optional<int64_t> maybeNewOffset = getConstantIntValue(offsets[i]);
    int64_t newStride = staticStrideVals[i];
    int64_t newSize = staticSizeVals[i];
    std::optional<int64_t> maybePrevOffset =
        getConstantIntValue(newOffsets[vecSize - 1]);
    int64_t prevStride = getConstantIndexOrAssert(newStrides[vecSize - 1]);
    int64_t prevSize = getConstantIndexOrAssert(newSizes[vecSize - 1]);
    int64_t dimExtent = prevStride * prevSize;
    // Fail if max constraints are provided, but the newly created
    // offsets/sizes/strides start exceeding the number of provide max
    // constraints as this will result in undefined behaviour.
    bool fitsMaxConstraint = checkValidSize(vecSize - 1, newSize * prevSize);
    if (fitsMaxConstraint && dimExtent == newStride) {
      // There are currently two cases supported for folding a dimension:
      // 1. If the offset is 0, we can fold the dimension, no matter what the
      // value of `newPrevOffset` is (it can be dynamic).
      // 2. If the offset, multiplied by the respective stride, is a multiple of
      // the previous stride, we can fold the dimension if we update the new
      // offset as well. However, in this case we need to add to new offset and
      // this is currently only supported for constant offsets.
      if (isConstantIntValue(offsets[i], 0)) {
        foldableLinearDimsFound = true;
        newSizes[vecSize - 1] = getAsIndexOpFoldResult(ctx, newSize * prevSize);
        continue;
      } else if (maybeNewOffset.has_value() && maybePrevOffset.has_value()) {
        // NOTE: It's guaranteed that
        // `(maybeNewOffset.value() * newStride) % prevStride == 0`
        // as `newStride == prevStride * prevSize`
        foldableLinearDimsFound = true;
        newSizes[vecSize - 1] = getAsIndexOpFoldResult(ctx, newSize * prevSize);
        int64_t newPrevOffset = maybePrevOffset.value() +
                                maybeNewOffset.value() * newStride / prevStride;
        newOffsets[vecSize - 1] = getAsIndexOpFoldResult(ctx, newPrevOffset);
        continue;
      }
    }
    newOffsets.push_back(offsets[i]);
    newStrides.push_back(strides[i]);
    newSizes.push_back(sizes[i]);
  }

  // Reverse as the new offsets/sizes/strides were created in reverse order.
  std::reverse(newOffsets.begin(), newOffsets.end());
  std::reverse(newSizes.begin(), newSizes.end());
  std::reverse(newStrides.begin(), newStrides.end());
  return success(foldableLinearDimsFound);
}

LogicalResult foldRepetitionCount(MLIRContext *ctx,
                                  SmallVector<OpFoldResult> &sizes,
                                  SmallVector<OpFoldResult> &strides,
                                  std::optional<int64_t> maybeRepetitionCount) {
  // If no repetition count is provided, fold all leading dimensions with
  // `stride == 0`.
  if (!maybeRepetitionCount.has_value()) {
    for (auto &&[idx, stride] : llvm::enumerate(strides)) {
      if (!isConstantIntValue(stride, 0)) break;
      sizes[idx] = getAsIndexOpFoldResult(ctx, 1);
    }
    return success();
  }
  int64_t repetitionCount = maybeRepetitionCount.value();
  for (size_t i = 0; i < strides.size(); i++) {
    if (repetitionCount == 1) break;
    if (!isConstantIntValue(strides[i], 0)) return failure();
    std::optional<int64_t> maybeSize = getConstantIntValue(sizes[i]);
    if (!maybeSize) return failure();
    int64_t size = maybeSize.value();
    if (size >= repetitionCount) {
      if (size % repetitionCount != 0) return failure();
      sizes[i] = getAsIndexOpFoldResult(ctx, size / repetitionCount);
      repetitionCount = 1;
    } else {
      if (repetitionCount % size != 0) return failure();
      sizes[i] = getAsIndexOpFoldResult(ctx, 1);
      repetitionCount /= size;
    }
  }
  if (repetitionCount != 1) return failure();
  return success();
}

/// Fold single dimension linear accesses and make them implicit. This requires
/// offset 0 and stride 1.
///
/// Example:
///   offsets: [%c0], sizes: [%c1024], strides: [%c1]
/// becomes:
///   offsets: [], sizes: [], strides: []
LogicalResult foldSingleDim(SmallVector<OpFoldResult> &offsets,
                            SmallVector<OpFoldResult> &sizes,
                            SmallVector<OpFoldResult> &strides) {
  assert(offsets.size() == sizes.size() && offsets.size() == strides.size() &&
         "expected same number of source offsets and sizes");

  if (offsets.size() != 1) return failure();
  if (!isConstantIntValue(offsets[0], 0)) return failure();
  if (!isConstantIntValue(strides[0], 1)) return failure();

  offsets.clear();
  sizes.clear();
  strides.clear();
  return success();
}

LogicalResult moveNpuDmaSyncUsersAfterAncestorInSameBlock(
    RewriterBase &rewriter, Operation *parentOp) {
  WalkResult res = parentOp->walk([&](AMDAIE::NpuDmaWaitOp npuDmaWaitOp) {
    SmallPtrSet<Operation *, 4> ancestorsInSameBlock;
    // All async token producers should result in the same ancestor being found.
    for (Value asyncToken : npuDmaWaitOp.getAsyncTokens()) {
      Operation *dmaOp = asyncToken.getDefiningOp();
      ancestorsInSameBlock.insert(
          getAncestorInBlock(npuDmaWaitOp, dmaOp->getBlock()));
    }
    if (ancestorsInSameBlock.size() == 0) {
      npuDmaWaitOp.emitOpError() << "no ancestors found";
      return WalkResult::interrupt();
    }
    if (ancestorsInSameBlock.size() != 1) {
      npuDmaWaitOp.emitOpError()
          << "the async token producers are located in a different scope";
      return WalkResult::interrupt();
    }
    Operation *ancestorInSameBlock = *ancestorsInSameBlock.begin();
    if (!ancestorInSameBlock) {
      npuDmaWaitOp->emitOpError(
          "doesn't have an ancestor in the same scope as the source DMA op");
      return WalkResult::interrupt();
    }
    rewriter.moveOpAfter(npuDmaWaitOp, ancestorInSameBlock);
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) return failure();
  return success();
}

/// Try to add `offsetToMerge` to one of the offsets in `offsets`.  This
/// function assumes an effective stride for `offsetToMerge` of 1 so to add
/// `offsetToMerge` to the offset in dimension `d`, `offsetToMerge` must be
/// divisable by the stride in dimension `d`.
///
/// \return false if `offsetToMerge` could not be added to any of the offsets.
bool mergeOffset(MLIRContext *ctx, int64_t offsetToMerge,
                 SmallVector<OpFoldResult> &offsets,
                 ArrayRef<OpFoldResult> strides) {
  if (offsetToMerge == 0) return true;
  for (uint32_t i = 0; i < offsets.size(); ++i) {
    std::optional<int64_t> cOffset = getConstantIntValue(offsets[i]);
    std::optional<int64_t> cStride = getConstantIntValue(strides[i]);
    if (cOffset.has_value() && cStride.has_value()) {
      int64_t offset = cOffset.value();
      int64_t stride = cStride.value();
      if (offsetToMerge % stride == 0) {
        offset += offsetToMerge / stride;
        offsets[i] = getAsIndexOpFoldResult(ctx, offset);
        return true;
      }
    }
  }
  return false;
}

/// This function tries to reduce the rank of the access pattern by merging
/// unit dimensions into other dimensions.
LogicalResult foldUnitDims(MLIRContext *ctx, SmallVector<OpFoldResult> &offsets,
                           SmallVector<OpFoldResult> &sizes,
                           SmallVector<OpFoldResult> &strides) {
  uint64_t initialRank = offsets.size();

  int64_t cumulativeOffset{0};
  int64_t insertionIndex{0};

  SmallVector<OpFoldResult> newOffsets;
  SmallVector<OpFoldResult> newSizes;
  SmallVector<OpFoldResult> newStrides;

  for (int i = 0; i < offsets.size(); ++i) {
    // If in dimension `i` there is constant offset, constant stride, and size
    // of 1, then update the cumulative offset.
    std::optional<int64_t> cOffset = getConstantIntValue(offsets[i]);
    std::optional<int64_t> cSize = getConstantIntValue(sizes[i]);
    std::optional<int64_t> cStride = getConstantIntValue(strides[i]);
    bool isSizeOne = cSize.has_value() && cSize.value() == 1;
    if (cOffset.has_value() && cStride.has_value() && isSizeOne) {
      cumulativeOffset += cOffset.value() * cStride.value();
    } else {
      insertionIndex += isSizeOne;
      newOffsets.push_back(offsets[i]);
      newSizes.push_back(sizes[i]);
      newStrides.push_back(strides[i]);
    }
  }

  // This is the case where there are no unit dimensions to fold.
  if (newStrides.size() == initialRank) return failure();

  bool merged = mergeOffset(ctx, cumulativeOffset, newOffsets, newStrides);

  // This is the case where there is one unit dimension, but it cannot be
  // merged into another dimension.
  if (!merged && (newStrides.size() + 1 == initialRank)) return failure();

  // At this point we know that we will be able to reduce the rank, and so will
  // start updating offsets, sizes, and strides.
  offsets = newOffsets;
  sizes = newSizes;
  strides = newStrides;
  if (!merged) {
    OpFoldResult one = getAsIndexOpFoldResult(ctx, 1);
    OpFoldResult offset = getAsIndexOpFoldResult(ctx, cumulativeOffset);
    offsets.insert(offsets.begin() + insertionIndex, one);
    sizes.insert(sizes.begin() + insertionIndex, one);
    strides.insert(strides.begin() + insertionIndex, offset);
  }

  assert(offsets.size() < initialRank && "Rank should have been reduced");
  return success();
}

//===----------------------------------------------------------------------===//
// DmaDimConfig
//===----------------------------------------------------------------------===//

static bool anyOutOfRange(ArrayRef<int64_t> values, ArrayRef<int64_t> maxValues,
                          size_t begin) {
  assert(maxValues.size() - begin >= values.size() &&
         "begin should be set so that the values don't exceed the max "
         "values slice");
  for (auto [value, maxValue] :
       llvm::zip(values, maxValues.drop_front(begin))) {
    if (value < 0 || value > maxValue) return true;
  }
  return false;
}

bool DmaDimConfig::isValidAccessPattern(ArrayRef<int64_t> sizes,
                                        ArrayRef<int64_t> strides) const {
  assert(sizes.size() == strides.size() &&
         "`sizes` and `strides` should have the same size");
  SmallVector<int64_t> maxSizes = getMaxSizes(sizes.size());
  assert(maxSizes.size() >= sizes.size() &&
         "Max number of dimensions exceeded");
  size_t frontToDrop = maxSizes.size() - sizes.size();
  if (anyOutOfRange(sizes, maxSizes, frontToDrop)) return false;
  SmallVector<int64_t> maxStrides = getMaxStrides(sizes.size());
  if (anyOutOfRange(strides, maxStrides, frontToDrop)) return false;
  return true;
}

SmallVector<int64_t> DmaDimConfig::getMaxSizes(
    std::optional<size_t> maybeNbDims) const {
  size_t nbDims = maybeNbDims.has_value() ? maybeNbDims.value() : maxNbDims;
  uint32_t maxIntraSize = deviceModel.getDmaBdProp<uint16_t>(
      tileType, 0, AMDAIE::AMDAIEDmaBdProp::WrapMax);
  uint32_t maxInterSize = deviceModel.getDmaBdProp<uint8_t>(
      tileType, 0, AMDAIE::AMDAIEDmaBdProp::IterWrapMax);
  SmallVector<int64_t> maxSizes(nbDims, 0);
  int64_t nbIntraDimsToBeFilled =
      std::min((int64_t)nbIntraDims, (int64_t)maxSizes.size());
  int64_t intraStart = maxSizes.size() - nbIntraDimsToBeFilled;
  std::fill_n(maxSizes.begin() + intraStart, nbIntraDimsToBeFilled,
              maxIntraSize);
  assert(intraStart >= 0 &&
         "The start index for intra dimensions should be greater than or equal "
         "to zero");
  if (intraStart < maxSizes.size())
    maxSizes[intraStart] = std::numeric_limits<int64_t>::max();
  int64_t nbInterDimsToBeFilled = std::min((int64_t)nbInterDims, intraStart);
  int64_t interStart = intraStart - nbInterDimsToBeFilled;
  assert(interStart >= 0 &&
         "The start index for inter dimensions should be greater than or equal "
         "to zero");
  std::fill_n(maxSizes.begin() + interStart, nbInterDimsToBeFilled,
              maxInterSize);
  return maxSizes;
}

SmallVector<int64_t> DmaDimConfig::getMaxStrides(
    std::optional<size_t> maybeNbDims) const {
  size_t nbDims = maybeNbDims.has_value() ? maybeNbDims.value() : maxNbDims;
  uint32_t maxIntraStride = deviceModel.getDmaBdProp<uint32_t>(
      tileType, 0, AMDAIE::AMDAIEDmaBdProp::StepSizeMax);
  uint32_t maxInterStride = deviceModel.getDmaBdProp<uint32_t>(
      tileType, 0, AMDAIE::AMDAIEDmaBdProp::IterStepSizeMax);
  SmallVector<int64_t> stepSizes(nbDims, 0);
  int64_t nbIntraDimsToBeFilled =
      std::min((int64_t)nbIntraDims, (int64_t)stepSizes.size());
  int64_t intraStart = stepSizes.size() - nbIntraDimsToBeFilled;
  assert(intraStart >= 0 &&
         "The start index for intra dimensions should be greater than or equal "
         "to zero");
  // +1 because values are encoded in HW BDs as (value - 1), so the range is
  // [1:2^x].
  std::fill_n(stepSizes.begin() + intraStart, nbIntraDimsToBeFilled,
              maxIntraStride + 1);
  int64_t nbInterDimsToBeFilled = std::min((int64_t)nbInterDims, intraStart);
  int64_t interStart = intraStart - nbInterDimsToBeFilled;
  assert(interStart >= 0 &&
         "The start index for inter dimensions should be greater than or equal "
         "to zero");
  // +1 because values are encoded in HW BDs as (value - 1), so the range is
  // [1:2^x].
  std::fill_n(stepSizes.begin() + interStart, nbInterDimsToBeFilled,
              maxInterStride + 1);
  return stepSizes;
}

SmallVector<int64_t> CircularDmaDimConfig::getMaxSizes(
    std::optional<size_t> maybeNbDims) const {
  size_t nbDims = maybeNbDims.has_value() ? maybeNbDims.value() : maxNbDims;
  uint32_t maxIntraSize = deviceModel.getDmaBdProp<uint16_t>(
      tileType, 0, AMDAIE::AMDAIEDmaBdProp::WrapMax);
  SmallVector<int64_t> maxSizes(nbDims, 0);
  int64_t nbIntraDimsToBeFilled =
      std::min((int64_t)nbIntraDims, (int64_t)maxSizes.size());
  int64_t intraStart = maxSizes.size() - nbIntraDimsToBeFilled;
  std::fill_n(maxSizes.begin() + intraStart, nbIntraDimsToBeFilled,
              maxIntraSize);
  assert(intraStart >= 0 &&
         "The start index for intra dimensions should be greater than or equal "
         "to zero");
  if (intraStart < maxSizes.size())
    maxSizes[intraStart] = std::numeric_limits<int64_t>::max();
  // All other dimension can have any size for circular DMAs.
  std::fill_n(maxSizes.begin(), intraStart,
              std::numeric_limits<int64_t>::max());
  return maxSizes;
}

SmallVector<int64_t> CircularDmaDimConfig::getMaxStrides(
    std::optional<size_t> maybeNbDims) const {
  size_t nbDims = maybeNbDims.has_value() ? maybeNbDims.value() : maxNbDims;
  uint32_t maxIntraStride = deviceModel.getDmaBdProp<uint32_t>(
      tileType, 0, AMDAIE::AMDAIEDmaBdProp::StepSizeMax);
  SmallVector<int64_t> stepSizes(nbDims, 0);
  int64_t nbIntraDimsToBeFilled =
      std::min((int64_t)nbIntraDims, (int64_t)stepSizes.size());
  int64_t intraStart = stepSizes.size() - nbIntraDimsToBeFilled;
  assert(intraStart >= 0 &&
         "The start index for intra dimensions should be greater than or equal "
         "to zero");
  // +1 because values are encoded in HW BDs as (value - 1), so the range is
  // [1:2^x].
  std::fill_n(stepSizes.begin() + intraStart, nbIntraDimsToBeFilled,
              maxIntraStride + 1);
  // All other dimension can have any stride for circular DMAs.
  std::fill_n(stepSizes.begin(), intraStart,
              std::numeric_limits<int64_t>::max());
  return stepSizes;
}

}  // namespace mlir::iree_compiler::AMDAIE
