// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIEDmaUtils.h"

#include "AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace mlir::iree_compiler::AMDAIE {

static bool isEqualConstantIntOrValueArrayFromIndices(
    ArrayRef<OpFoldResult> ofrsA, ArrayRef<OpFoldResult> ofrsB,
    size_t indexA = 0, size_t indexB = 0) {
  if ((ofrsA.size() - indexA) != (ofrsB.size() - indexB)) return false;
  return isEqualConstantIntOrValueArray(ofrsA.drop_front(indexA),
                                        ofrsB.drop_front(indexB));
}

bool areAccessPatternsEqualFromIndices(ArrayRef<OpFoldResult> offsetsA,
                                       ArrayRef<OpFoldResult> sizesA,
                                       ArrayRef<OpFoldResult> stridesA,
                                       ArrayRef<OpFoldResult> offsetsB,
                                       ArrayRef<OpFoldResult> sizesB,
                                       ArrayRef<OpFoldResult> stridesB,
                                       size_t indexA, size_t indexB) {
  return isEqualConstantIntOrValueArrayFromIndices(offsetsA, offsetsB, indexA,
                                                   indexB) &&
         isEqualConstantIntOrValueArrayFromIndices(sizesA, sizesB, indexA,
                                                   indexB) &&
         isEqualConstantIntOrValueArrayFromIndices(stridesA, stridesB, indexA,
                                                   indexB);
}

bool areAccessPatternsCombinable(const SmallVector<OpFoldResult> &offsetsA,
                                 const SmallVector<OpFoldResult> &sizesA,
                                 const SmallVector<OpFoldResult> &stridesA,
                                 const SmallVector<OpFoldResult> &offsetsB,
                                 const SmallVector<OpFoldResult> &sizesB,
                                 const SmallVector<OpFoldResult> &stridesB,
                                 size_t maxNbDims) {
  assert(offsetsA.size() == sizesA.size() &&
         "expected same number of source offsets and sizes");
  assert(offsetsA.size() == stridesA.size() &&
         "expected same number of source offsets and strides");
  assert(offsetsB.size() == sizesB.size() &&
         "expected same number of source offsets and sizes");
  assert(offsetsB.size() == stridesB.size() &&
         "expected same number of source offsets and strides");
  if (std::abs((ssize_t)offsetsA.size() - (ssize_t)offsetsB.size()) > 1)
    return false;
  // Empty access patterns are always combinable as they effectively mean:
  // 'don't perform any or change the addressing'.
  if (offsetsA.empty() && offsetsB.empty()) return true;
  // In case both access patterns have the same number of dimension, a new
  // dimension will need to be added, so fail if there aren't enough
  // dimensions.
  if (offsetsA.size() == offsetsB.size() && offsetsA.size() + 1 > maxNbDims)
    return false;

  // Equality of the last N elements of the access patterns of A and B with N =
  // min(sizeA, sizeB) results in some simple cases in which the access
  // patterns are combinable. Note that abs(sizeB - sizeA) should <= 1 and this
  // is checked for earlier, so just asserted here.
  assert(std::abs((ssize_t)offsetsB.size() - (ssize_t)offsetsA.size()) <= 1 &&
         "The distance between the indices should be smaller or equal to one.");
  size_t indexA = offsetsA.size() > offsetsB.size() ? 1 : 0;
  size_t indexB = offsetsB.size() > offsetsA.size() ? 1 : 0;
  if (areAccessPatternsEqualFromIndices(offsetsA, sizesA, stridesA, offsetsB,
                                        sizesB, stridesB, indexA, indexB)) {
    if (offsetsA.size() == offsetsB.size()) {
      return true;
    } else if (offsetsA.size() > offsetsB.size()) {
      // The access pattern A has N repetitions of access pattern B, so they can
      // be combined together into N+1 repetitions.
      return isConstantIntValue(stridesA[0], 0);
    } else {
      // offsetsB.size() > offsetsA.size()
      // The access pattern B has N repetitions of access pattern A, so they can
      // be combined together into N+1 repetitions.
      if (isConstantIntValue(stridesB[0], 0)) return true;
      // The access pattern of B is the same as the access pattern of A, but at
      // a different offset. They can be combined by reducing the offset of B to
      // zero.
      if (isConstantIntValue(offsetsB[0], 1)) return true;
      return false;
    }
  }

  for (auto &&[strideA, strideB] :
       llvm::zip(llvm::reverse(stridesA), llvm::reverse(stridesB))) {
    std::optional<int64_t> maybeStrideA = getConstantIntValue(strideA);
    std::optional<int64_t> maybeStrideB = getConstantIntValue(strideB);
    // Handle static and constant value with same int value.
    if (maybeStrideA && maybeStrideB &&
        maybeStrideA.value() == maybeStrideB.value()) {
      continue;
    }
    if (strideA != strideB) return false;
  }

  // Don't check the outermost dimension of size at this point.
  SmallVector<OpFoldResult> innerSizesA;
  SmallVector<OpFoldResult> innerSizesB;
  std::copy(sizesA.begin() + 1, sizesA.end(), std::back_inserter(innerSizesA));
  std::copy(sizesB.begin() + 1, sizesB.end(), std::back_inserter(innerSizesB));
  for (auto &&[sizeA, sizeB] :
       llvm::zip(llvm::reverse(innerSizesA), llvm::reverse(innerSizesB))) {
    std::optional<int64_t> maybeSizeA = getConstantIntValue(sizeA);
    std::optional<int64_t> maybeSizeB = getConstantIntValue(sizeB);
    // Handle static and constant value with same int value.
    if (maybeSizeA && maybeSizeB && maybeSizeA.value() == maybeSizeB.value()) {
      continue;
    }
    if (sizeA != sizeB) return false;
  }

  // Edge case for sizesA[0] != sizesB[0].
  if (offsetsB.size() == offsetsA.size() && sizesA[0] != sizesB[0]) {
    std::optional<int64_t> constOffsetA = getConstantIntValue(offsetsA[0]);
    std::optional<int64_t> constSizeA = getConstantIntValue(sizesA[0]);
    std::optional<int64_t> constOffsetB = getConstantIntValue(offsetsB[0]);
    std::optional<int64_t> constSizeB = getConstantIntValue(sizesB[0]);
    if (constOffsetA && constOffsetB && constSizeA && constSizeB) {
      int64_t offsetDiff = constOffsetB.value() - constOffsetA.value();
      if (constSizeA.value() != offsetDiff) return false;
    } else {
      return false;
    }
  }

  bool foundDiff{false};
  for (auto iter : llvm::enumerate(
           llvm::zip(llvm::reverse(offsetsA), llvm::reverse(offsetsB)))) {
    const OpFoldResult &offsetA = std::get<0>(iter.value());
    const OpFoldResult &offsetB = std::get<1>(iter.value());
    if (offsetA == offsetB) continue;
    std::optional<int64_t> maybeOffsetA = getConstantIntValue(offsetA);
    std::optional<int64_t> maybeOffsetB = getConstantIntValue(offsetB);
    if (maybeOffsetA && maybeOffsetB &&
        maybeOffsetA.value() == maybeOffsetB.value()) {
      continue;
    }
    // Retrieve the corresponding stride for this dimension.
    std::optional<int64_t> maybeStride =
        getConstantIntValue(stridesA[stridesA.size() - 1 - iter.index()]);
    if (maybeOffsetA && maybeOffsetB && maybeStride) {
      int64_t diff =
          (maybeOffsetB.value() - maybeOffsetA.value()) * maybeStride.value();
      // Handle the three different size cases. Return early in case of an
      // incompatibility.
      if (offsetsA.size() > offsetsB.size()) {
        std::optional<int64_t> constOffset = getConstantIntValue(offsetsA[0]);
        std::optional<int64_t> constStride = getConstantIntValue(stridesA[0]);
        std::optional<int64_t> constSize = getConstantIntValue(sizesA[0]);
        if (constOffset && constStride && constSize &&
            constOffset.value() == 0 &&
            (constStride.value() * constSize.value()) == diff) {
          if (foundDiff) return false;
          foundDiff = true;
        } else {
          return false;
        }
      } else if (offsetsB.size() > offsetsA.size()) {
        std::optional<int64_t> constOffset = getConstantIntValue(offsetsB[0]);
        std::optional<int64_t> constStride = getConstantIntValue(stridesB[0]);
        if (constOffset && constStride && constOffset.value() == 0 &&
            constStride.value() == diff) {
          if (foundDiff) return false;
          foundDiff = true;
        } else {
          return false;
        }
      } else {
        if (foundDiff) return false;
        foundDiff = true;
      }
    } else {
      return false;
    }
  }
  return foundDiff;
}

LogicalResult combineAccessPatterns(RewriterBase &rewriter,
                                    const SmallVector<OpFoldResult> &offsetsA,
                                    const SmallVector<OpFoldResult> &sizesA,
                                    const SmallVector<OpFoldResult> &stridesA,
                                    const SmallVector<OpFoldResult> &offsetsB,
                                    const SmallVector<OpFoldResult> &sizesB,
                                    const SmallVector<OpFoldResult> &stridesB,
                                    SmallVector<OpFoldResult> &newOffsets,
                                    SmallVector<OpFoldResult> &newSizes,
                                    SmallVector<OpFoldResult> &newStrides,
                                    size_t maxNbDims) {
  assert(offsetsA.size() == sizesA.size() &&
         "expected same number of source offsets and sizes");
  assert(offsetsA.size() == stridesA.size() &&
         "expected same number of source offsets and strides");
  assert(offsetsB.size() == sizesB.size() &&
         "expected same number of source offsets and sizes");
  assert(offsetsB.size() == stridesB.size() &&
         "expected same number of source offsets and strides");
  if (!areAccessPatternsCombinable(offsetsA, sizesA, stridesA, offsetsB, sizesB,
                                   stridesB, maxNbDims)) {
    return failure();
  }
  if (offsetsA.empty() && offsetsB.empty()) return success();
  if (offsetsB.size() > offsetsA.size()) {
    newOffsets = offsetsB;
    newSizes = sizesB;
    newStrides = stridesB;
    // If the offset on the first dimension of B is larger than zero, we can
    // just decrease that one by one to accomplish the access pattern merge.
    // Otherwise, we check for and update the other differing offsets.
    std::optional<int64_t> offset = getConstantIntValue(newOffsets[0]);
    if (offset && offset.value() > 0) {
      newOffsets[0] = rewriter.getI64IntegerAttr(offset.value() - 1);
    } else {
      for (int i = 1; i <= offsetsA.size(); i++) {
        if (offsetsA[offsetsA.size() - i] != offsetsB[offsetsB.size() - i]) {
          newOffsets[newOffsets.size() - i] = offsetsA[offsetsA.size() - i];
          break;
        }
      }
    }
    std::optional<int64_t> size = getConstantIntValue(newSizes[0]);
    if (!size) return failure();
    newSizes[0] = rewriter.getI64IntegerAttr(size.value() + 1);
  } else if (offsetsA.size() > offsetsB.size()) {
    newOffsets = offsetsA;
    newSizes = sizesA;
    newStrides = stridesA;
    std::optional<int64_t> size = getConstantIntValue(newSizes[0]);
    if (!size) return failure();
    newSizes[0] = rewriter.getI64IntegerAttr(size.value() + 1);
  } else {
    // Edge case for sizesA[0] != sizesB[0].
    if (sizesA[0] != sizesB[0]) {
      newOffsets = offsetsA;
      newSizes = sizesA;
      newStrides = stridesA;
      std::optional<int64_t> sizeA = getConstantIntValue(sizesA[0]);
      std::optional<int64_t> sizeB = getConstantIntValue(sizesB[0]);
      if (!sizeA || !sizeB) return failure();
      newSizes[0] = rewriter.getI64IntegerAttr(sizeA.value() + sizeB.value());
    } else {
      // All dims of sizes are the same, so add a new dimension with
      // 'offset == 0', 'size == 2' and 'stride == offsetDiff'.
      newOffsets.push_back(rewriter.getI64IntegerAttr(0));
      int64_t offsetDiff{0};
      int64_t strideMultiplier{0};
      for (auto iter : llvm::enumerate(llvm::zip(offsetsA, offsetsB))) {
        const OpFoldResult &offsetA = std::get<0>(iter.value());
        const OpFoldResult &offsetB = std::get<1>(iter.value());
        newOffsets.push_back(offsetA);
        if (offsetA != offsetB) {
          std::optional<int64_t> constOffsetA = getConstantIntValue(offsetA);
          std::optional<int64_t> constOffsetB = getConstantIntValue(offsetB);
          if (!constOffsetA || !constOffsetB) {
            return emitError(rewriter.getUnknownLoc())
                   << "differing offsets should be constants";
          }
          offsetDiff = constOffsetB.value() - constOffsetA.value();
          std::optional<int64_t> maybeStride =
              getConstantIntValue(stridesA[iter.index()]);
          if (!maybeStride) {
            return emitError(rewriter.getUnknownLoc())
                   << "no constant stride found at the same index where the "
                      "offset "
                      "difference occurs";
          }
          strideMultiplier = maybeStride.value();
        }
      }
      newSizes.push_back(rewriter.getI64IntegerAttr(2));
      newSizes.append(sizesA.begin(), sizesA.end());
      newStrides.push_back(
          rewriter.getI64IntegerAttr(offsetDiff * strideMultiplier));
      newStrides.append(stridesA.begin(), stridesA.end());
    }
  }
  assert(newOffsets.size() == newSizes.size() &&
         "expected same number of new offsets and sizes");
  assert(newOffsets.size() == newStrides.size() &&
         "expected same number of new offsets and strides");
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
    // 1. Offsets[i] == 0.This is required because we are dropping the offset
    // of the i dimension and keep newOffets[-1]
    // 2. newSizes[-1] x newStrides[-1] == strides[i]. With this we can have
    // newSizes[-1] = sizes[i] * newSizes[-1] , and then fold away the i
    // dimension
    // 3. checkValidSize(sizes[i] * newSizes[-1]). This allows hardware
    // constraints to be checked.
    size_t vecSize = newOffsets.size();
    int64_t newStride = staticStrideVals[i];
    int64_t newSize = staticSizeVals[i];
    int64_t prevStride = getConstantIndexOrAssert(newStrides[vecSize - 1]);
    int64_t prevSize = getConstantIndexOrAssert(newSizes[vecSize - 1]);
    int64_t dimExtent = prevStride * prevSize;
    // Fail if max constraints are provided, but the newly created
    // offsets/sizes/strides start exceeding the number of provide max
    // constraints as this will result in undefined behaviour.
    bool fitsMaxConstraint = checkValidSize(vecSize - 1, newSize * prevSize);
    if (fitsMaxConstraint && isConstantIntValue(offsets[i], 0) &&
        dimExtent == newStride) {
      foldableLinearDimsFound = true;
      newSizes[vecSize - 1] = getAsIndexOpFoldResult(ctx, newSize * prevSize);
      continue;
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

/// Fold unit dimensions within a strided access pattern. There are two cases
/// being handled here:
/// 1. If a dimension has `size == 1` and `offset == 0`, the dimension can be
/// folded entirely.
/// 2. If a dimension has `size == 1` and `offset != 0`, it can be folded into
/// another dimension with the same stride if that exists.
LogicalResult foldUnitDims(MLIRContext *ctx,
                           const SmallVector<OpFoldResult> &offsets,
                           const SmallVector<OpFoldResult> &sizes,
                           const SmallVector<OpFoldResult> &strides,
                           SmallVector<OpFoldResult> &newOffsets,
                           SmallVector<OpFoldResult> &newSizes,
                           SmallVector<OpFoldResult> &newStrides) {
  bool foldableUnitDimsFound = false;
  DenseMap<int64_t, std::pair<size_t, int64_t>> strideToIndexAndOffset;
  for (int i = 0; i < offsets.size(); i++) {
    // If a dimension has `size == 1` and `offset == 0`, the dimension can be
    /// folded entirely.
    if (isConstantIntValue(offsets[i], 0) && isConstantIntValue(sizes[i], 1)) {
      foldableUnitDimsFound = true;
      continue;
    }
    std::optional<int64_t> maybeOffset = getConstantIntValue(offsets[i]);
    std::optional<int64_t> maybeStride = getConstantIntValue(strides[i]);
    if (maybeOffset && maybeStride) {
      int64_t offset = maybeOffset.value();
      int64_t stride = maybeStride.value();
      if (isConstantIntValue(sizes[i], 1) &&
          strideToIndexAndOffset.contains(stride)) {
        foldableUnitDimsFound = true;
        strideToIndexAndOffset[stride].second += offset;
        // Continue to not add to newOffsets, newSizes, newStrides
        continue;
      } else {
        strideToIndexAndOffset[stride] = {newOffsets.size(), offset};
      }
    }
    newOffsets.push_back(offsets[i]);
    newStrides.push_back(strides[i]);
    newSizes.push_back(sizes[i]);
  }
  // Update offsets
  for (auto &&[stride, indexAndOffset] : strideToIndexAndOffset) {
    newOffsets[indexAndOffset.first] =
        getAsIndexOpFoldResult(ctx, indexAndOffset.second);
  }
  return success(foldableUnitDimsFound);
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
