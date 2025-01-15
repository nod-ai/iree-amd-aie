// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEDMAUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEDMAUTILS_H_

#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::AMDAIE {

/// Utility affine expression visitor to retrieve the scale and optional bias
/// from the expression.
struct RetrieveScaleAndBias
    : public AffineExprVisitor<RetrieveScaleAndBias, LogicalResult> {
  std::optional<int64_t> scale;
  std::optional<int64_t> bias;
  LogicalResult visitAffineBinaryOpExpr(AffineBinaryOpExpr /*expr*/) {
    return failure();
  }
  LogicalResult visitConstantExpr(AffineConstantExpr /*expr*/) {
    return failure();
  }
  LogicalResult visitDimExpr(AffineDimExpr /*expr*/) { return failure(); }
  LogicalResult visitSymbolExpr(AffineSymbolExpr /*expr*/) { return failure(); }
  LogicalResult visitMulExpr(AffineBinaryOpExpr expr) {
    if (auto rhsSize = dyn_cast<AffineConstantExpr>(expr.getRHS());
        isa<AffineDimExpr>(expr.getLHS())) {
      scale = rhsSize.getValue();
    } else if (auto lhsSize = dyn_cast<AffineConstantExpr>(expr.getLHS());
               isa<AffineDimExpr>(expr.getRHS())) {
      scale = lhsSize.getValue();
    }
    return success();
  }
  LogicalResult visitAddExpr(AffineBinaryOpExpr expr) {
    if (bias) return failure();
    if (auto rhsSize = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      bias = rhsSize.getValue();
      if (bias.value() < 0) return failure();
      if (isa<AffineBinaryOpExpr>(expr.getLHS())) {
        return visit(expr.getLHS());
      } else if (isa<AffineDimExpr>(expr.getLHS())) {
        scale = 1;
        return success();
      } else {
        return failure();
      }
    } else if (auto lhsSize = dyn_cast<AffineConstantExpr>(expr.getLHS())) {
      bias = lhsSize.getValue();
      if (bias.value() < 0) return failure();
      if (isa<AffineBinaryOpExpr>(expr.getRHS())) {
        return visit(expr.getRHS());
      } else if (isa<AffineDimExpr>(expr.getRHS())) {
        scale = 1;
        return success();
      } else {
        return failure();
      }
    } else {
      return failure();
    }
  }
};

namespace detail {

/// Update the strides and offsets of `X` to match the strides of `Y` if it is
/// possible to do so without changing the underlying access pattern of `X`. For
/// example if
///
/// X has access pattern (offset: [5] sizes: [1] strides: [6]) and
/// Y has access pattern (offset: [a] sizes: [b] strides: [3])
///
/// Then the access pattern for X can be changed to have access pattern
/// (offset: [10] sizes: [1] strides: [3]) so that its stride matches Y's.
///
/// For this transformation to be possible in dimension `d` is it necessary that
///
/// 1) the size in dimension `d` of `X` is 1, and
/// 2) the updated offset in `d` of `X` (i.e. offset * strideX / strideY)
///    is an integer.
///
/// As another example, if we have:
///
/// X with access pattern (offset: [4, 8]  sizes: [1, 1] strides: [3, 8])
/// Y with access pattern (offset: [a, b]  sizes: [d, e] strides: [6, 2])
///
/// then X can be transformed to have access pattern
///                       (offset: [2, 32] sizes: [1, 1] strides: [6, 2])
void matchStridesOfUnitDims(MLIRContext *ctx, ArrayRef<OpFoldResult> sizesX,
                            SmallVector<OpFoldResult> &stridesX,
                            SmallVector<OpFoldResult> &offsetsX,
                            ArrayRef<OpFoldResult> stridesY);

/// This function computes the difference between the global offsets of two
/// access patterns. If it is not constant, i.e. if the difference contains
/// an MLIR value which is not a constant, then nullopt is returned.
///
/// This function is useful when determining if the access pattern A, followed
/// by the access pattern B, can be merged into a single access pattern.
///
/// The reason for this API design, as opposed to a more intuitive design of
/// having a function to compute the global offset difference for a single
/// access pattern, say `getGlobalOffset`, and then computing the difference
/// as `getGlobalOffset(A) - getGlobalOffset(B)`, is that the global offset
/// difference might be constant while each individual offsets are not, and
/// determining that the non-constants cancel is easier with this API.
///
/// \return global_offset(X) - global_offset(Y).
///
/// Background info: offsets, sizes, and strides define an access pattern into
/// an array, where the i'th element accessed, for 0 <= i < prod_{d<D} sizes[d],
/// is at index
///
///     sum_{d<D} (l(d,i) + offset[d]) * stride[d]                (1)
///
/// where l(d,i) is the component of global index `i` in dimension `d`:
///
///     i  = sum_{d<D} l(d,i) * size[d]                           (2)
///
/// Equation (1) can be rewritten with a global offset as
///
///     global_offset + sum_{d<D} l(d,i) * stride[d]              (3)
///
/// where the global offset is
///
///     global_offset = sum_{d<D} offset[d] * stride[d].
///
std::optional<int64_t> getGlobalOffsetDifference(
    ArrayRef<OpFoldResult> offsetsX, ArrayRef<OpFoldResult> stridesX,
    ArrayRef<OpFoldResult> offsetsY, ArrayRef<OpFoldResult> stridesY);

}  // namespace detail

/// Combine two access patterns into a single one. Assumes that access pattern A
/// belongs to a strided op which is ordered before the strided op B. Takes a
/// `maxNbDims` argument to ensure that a combined access pattern would not
/// exceed the maximum number of dimensions. Returns `success` if the access
/// patterns were combined successfully.
LogicalResult combineAccessPatterns(
    MLIRContext *, ArrayRef<OpFoldResult> offsetsA,
    ArrayRef<OpFoldResult> sizesA, ArrayRef<OpFoldResult> stridesA,
    ArrayRef<OpFoldResult> offsetsB, ArrayRef<OpFoldResult> sizesB,
    ArrayRef<OpFoldResult> stridesB, SmallVector<OpFoldResult> &newOffsets,
    SmallVector<OpFoldResult> &newSizes, SmallVector<OpFoldResult> &newStrides,
    function_ref<bool(size_t)> exceedsNbDims);

/// Fold subsequent dimensions within a strided access pattern that describe a
/// single linear access. Returns `success` if folding took place.
/// Accepts optional maximum size constraints as an array of integers. Note that
/// `maxSizes` is expected to be provided in the same order as `sizes` and they
/// are compared from right to left (innermost to outermost). Also note that the
/// number of max sizes might exceed the number of sizes and the other way
/// around, BUT after canonicalization, the number of sizes should be smaller or
/// equal to the number of max sizes (if specified).
LogicalResult foldLinearDims(
    MLIRContext *, const SmallVector<OpFoldResult> &offsets,
    const SmallVector<OpFoldResult> &sizes,
    const SmallVector<OpFoldResult> &strides,
    SmallVector<OpFoldResult> &newOffsets, SmallVector<OpFoldResult> &newSizes,
    SmallVector<OpFoldResult> &newStrides,
    function_ref<bool(size_t, int64_t)> checkValidSize =
        [](size_t idxFromEnd, int64_t size) { return true; });

/// Utility to fold a provided repetition count from the front of the access
/// pattern (dimensions with `size > 1` and `stride == 0` indicate a
/// repetition). This function fails if the access pattern doesn't have the
/// provided repetition count at the front.
///
/// This transformation is useful for circular DMA operations that don't need
/// the repetition count to be specified. Note however that not all repetition
/// dimensions are necessarily completely removed as that could potentially
/// invalidate the DMA operation as part of the repetition could be operating on
/// the same data slice, while part could be operating on a new data slice. This
/// function doesn't make any assumption on that, but that's the reason why an
/// optional repetition count can be passed.
///
/// Example:
///
/// sizes: [32, 8], strides: [0, 1], repetitionCount: 16
///
/// will be transformed into:
///
/// sizes: [2, 8], strides: [0, 1]
LogicalResult foldRepetitionCount(
    MLIRContext *, SmallVector<OpFoldResult> &sizes,
    SmallVector<OpFoldResult> &strides,
    std::optional<int64_t> maybeRepetitionCount = std::nullopt);

/// Fold single dimension linear accesses and make them implicit. `This
/// operation happens in place. Returns `success` if folding took place.
LogicalResult foldSingleDim(SmallVector<OpFoldResult> &offsets,
                            SmallVector<OpFoldResult> &sizes,
                            SmallVector<OpFoldResult> &strides);

/// Fold unit dimensions within a strided access pattern. Returns `success` if
/// folding took place. If `success` is returned, then the rank of the access
/// pattern has been reduced. If `failure` is returned, `offsets`, `sizes`, and
/// `strides` are left unchanged.
///
/// Unit dimensions without any offset can be directly removed. Other unit
/// dimensions with offsets contribute to a global offset, which can be
/// considered an 'initial pointer address'.
///
/// The algorithm works roughly as follows:
/// (1) find all the unit dimensions with constant stride and offset and combine
///     them into a single dimension.
/// (2) try to merge the offset from step (1) into a non-unit dimension. This
///     requires finding a non-unit dimension with a stride that divides the
///     offset in (1). This is guaranteed to succeed if there is dimension with
///     stride 1, which is usually the case for the inner-most dimension.
///
///
/// After this function has been called, `sizes` will contain either one or
/// zero unit dimensions `d` where offset[d] and stride[d] are constant. It will
/// contain zero such dimensions if step (2) above was successful, and one if
/// it was not.
///
/// Example 1:
/// ---------
///
///   offsets: [0, 0, 0], sizes: [32, 1, 8], strides: [32, 1024, 1]
///
/// this has a global offset of 0, and will be transformed into:
///
///   offsets: [0, 0], sizes: [32, 8], strides: [32, 1]
///
/// Example 2:
/// ---------
///
///   offsets: [1, 0, 1, 0], sizes: [1, 32, 1, 8], strides: [1024, 32, 1024, 1]
///
/// this has a global offset of 2048, and will be transformed into:
///
///   offsets: [64, 0], sizes: [32, 8], strides: [32, 1]
///
/// it could equally well have been transformed into
///
///   offsets: [0, 2048], sizes: [32, 8], strides: [32, 1]
///
/// but the current implementation arbitrarily attempts to be merge the offset
/// starting from the left-most dimension.
///
/// Example 3:
/// ---------
///
///   offsets: [2, 2, 15],  sizes: [1, 1, 10],  strides: [4, 6, 10]
///
/// becomes
///
///   offset: [17],  sizes: [10], strides: [10]
///
/// Example 4:
/// ---------
///
///   offset: [3, 1, 15],  sizes: [1, 1, 10],  strides: [4, 6, 10]
///
/// becomes
///
///   offset: [1, 15],  sizes: [1, 10],  strides: [18, 10]
///
/// In this example, step (2) of the algorithm failed, but `success` is still
/// returned because the rank of the access pattern was reduced.
LogicalResult foldUnitDims(MLIRContext *, SmallVector<OpFoldResult> &offsets,
                           SmallVector<OpFoldResult> &strides,
                           SmallVector<OpFoldResult> &sizes);

/// Utility DMA configuration which is calculated based on AMDAIEDeviceModel
/// information.
///
/// Context:
/// Depending on the DMA being targetted, there can be a different
/// number of max dimensions supported by the hardware. Consider the different
/// cases for Shim, MemTile and core DMAs:
/// - Shim DMAs: As the shim DMA typically isn't synchronized with other DMAs
///   (through semaphore locks), the inter-iteration access pattern is
///   typically used as an additional intra-iteration access pattern,
///   resulting in 4 DMA dimensions which can be used to address global
///   memory data.
/// - As the MemTile DMAs are typically synchronized with other DMAs for
///   stream-through, double-buffering purposes, the inter-iteration can't
///   typically be used in the same way as the intra-iteration dimensions.
///   Therefore, for now, only the intra-iteration dimensions can be used for
///   DMA access patterns.
/// - Core DMAs: As the core DMAs are typically synchronized with the core
///   processor for data access purposes (read/write), the inter-iteration
///   can't typically be used in the same way as the intra-iteration
///   dimensions. Therefore, for now, only the intra-iteration dimensions can
///   be used for DMA access patterns.
struct DmaDimConfig {
  const AMDAIE::AMDAIEDeviceModel &deviceModel;
  AMDAIE::AMDAIETileType tileType;
  /// The maximum number of addressing dimensions on of the DMA.
  uint8_t maxNbDims{0};
  /// The number of `inter` addressing dimensions on of the DMA.
  uint8_t nbInterDims{0};
  /// The number of `intra` addressing dimensions on of the DMA.
  uint8_t nbIntraDims{0};

  DmaDimConfig(const AMDAIE::AMDAIEDeviceModel &deviceModel, uint8_t memSpace)
      : deviceModel(deviceModel) {
    if (memSpace == 0) {
      nbIntraDims = deviceModel.getDmaProp<uint8_t>(
          AMDAIE::AMDAIETileType::SHIMNOC, AMDAIE::AMDAIEDmaProp::NumAddrDim);
      tileType = AMDAIE::AMDAIETileType::SHIMNOC;
      nbInterDims = deviceModel.deviceConfig.dmaNbInterDims;
      maxNbDims = nbIntraDims + nbInterDims;
    } else if (memSpace == 1) {
      nbIntraDims = deviceModel.getDmaProp<uint8_t>(
          AMDAIE::AMDAIETileType::MEMTILE, AMDAIE::AMDAIEDmaProp::NumAddrDim);
      tileType = AMDAIE::AMDAIETileType::MEMTILE;
      maxNbDims = nbIntraDims;
    } else if (memSpace == 2) {
      nbIntraDims = deviceModel.getDmaProp<uint8_t>(
          AMDAIE::AMDAIETileType::AIETILE, AMDAIE::AMDAIEDmaProp::NumAddrDim);
      tileType = AMDAIE::AMDAIETileType::AIETILE;
      maxNbDims = nbIntraDims;
    } else {
      assert(false && "unsupported memspace: ");
    }
  }
  virtual ~DmaDimConfig(){};

  bool isValidAccessPattern(SmallVector<int64_t> sizes,
                            SmallVector<int64_t> strides) const;

  /// Return a vector containing the max size values for every dimension.
  virtual SmallVector<int64_t> getMaxSizes(
      std::optional<size_t> maybeNbDims = std::nullopt) const;

  /// Return a vector containing the max stride values for every dimension.
  virtual SmallVector<int64_t> getMaxStrides(
      std::optional<size_t> maybeNbDims = std::nullopt) const;

  virtual bool exceedsNbDims(size_t dims) const { return dims > maxNbDims; }
};

/// Contains utility DMA information for circular DMA operations which is
/// calculated based on AMDAIEDeviceModel information.
struct CircularDmaDimConfig final : public DmaDimConfig {
  CircularDmaDimConfig(const AMDAIE::AMDAIEDeviceModel &deviceModel,
                       uint8_t memSpace)
      : DmaDimConfig(deviceModel, memSpace) {}

  SmallVector<int64_t> getMaxSizes(
      std::optional<size_t> maybeNbDims = std::nullopt) const;

  SmallVector<int64_t> getMaxStrides(
      std::optional<size_t> maybeNbDims = std::nullopt) const;

  bool exceedsNbDims(size_t dims) const {
    // Allow any number of dimensions for circular DMAs.
    return false;
  }
};

/// Utility to move the synchronization users (`amdaie.npu.dma_wait`) directly
/// after its ancestor in the same block as the DMA operation it's synchronizing
/// on. This utility can be used for cleanup after DMA transformations to avoid
/// deadlocks and/or ensure SSA dominance. The idea is to ensure correct
/// synchronization by not influencing whatever is happening in between the
/// async DMA operation and its synchronization op.
LogicalResult moveNpuDmaSyncUsersAfterAncestorInSameBlock(
    RewriterBase &rewriter, Operation *parentOp);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
