// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEDMAUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEDMAUTILS_H_

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEDmaOpInterface.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

/// Check whether two access patterns are equal in value, starting from
/// specified indices.
bool areAccessPatternsEqualFromIndices(ArrayRef<OpFoldResult> offsetsA,
                                       ArrayRef<OpFoldResult> sizesA,
                                       ArrayRef<OpFoldResult> stridesA,
                                       ArrayRef<OpFoldResult> offsetsB,
                                       ArrayRef<OpFoldResult> sizesB,
                                       ArrayRef<OpFoldResult> stridesB,
                                       size_t indexA = 0, size_t indexB = 0);

/// Check whether the two access patterns of strided operations can be combined
/// into one. Takes a `maxNbDims` argument to check whether a combined access
/// pattern would not exceed the maximum number of dimensions.
bool areAccessPatternsCombinable(const SmallVector<OpFoldResult> &offsetsA,
                                 const SmallVector<OpFoldResult> &sizesA,
                                 const SmallVector<OpFoldResult> &stridesA,
                                 const SmallVector<OpFoldResult> &offsetsB,
                                 const SmallVector<OpFoldResult> &sizesB,
                                 const SmallVector<OpFoldResult> &stridesB,
                                 size_t maxNbDims);

/// Combine two access patterns into a single one. Assumes that access pattern A
/// belongs to a strided op which is ordered before the strided op B. Takes a
/// `maxNbDims` argument to ensure that a combined access pattern would not
/// exceed the maximum number of dimensions. Returns `success` if the access
/// patterns were combined successfully.
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
                                    size_t maxNbDims);

/// Fold subsequent dimensions within a strided access pattern that describe a
/// single linear access. Returns `success` if folding took place.
/// Accepts optional maximum size constraints as an array of integers. Note that
/// `maxSizes` is expected to be provided in the same order as `sizes` and they
/// are compared from right to left (innermost to outermost). Also note that the
/// number of max sizes might exceed the number of sizes and the other way
/// around, BUT after canonicalization, the number of sizes should be smaller or
/// equal to the number of max sizes (if specified).
LogicalResult foldLinearDims(MLIRContext *ctx,
                             const SmallVector<OpFoldResult> &offsets,
                             const SmallVector<OpFoldResult> &sizes,
                             const SmallVector<OpFoldResult> &strides,
                             SmallVector<OpFoldResult> &newOffsets,
                             SmallVector<OpFoldResult> &newSizes,
                             SmallVector<OpFoldResult> &newStrides,
                             ArrayRef<int64_t> maxSizes = {});

/// Fold single dimension linear accesses and make them implicit. `This
/// operation happens in place. Returns `success` if folding took place.
LogicalResult foldSingleDim(SmallVector<OpFoldResult> &offsets,
                            SmallVector<OpFoldResult> &sizes,
                            SmallVector<OpFoldResult> &strides);

/// Fold unit dimensions within a strided access pattern. Returns `success` if
/// folding took place. There are two cases being handled here:
/// 1. If a dimension has `size == 1` and `offset == 0`, the dimension can be
/// folded entirely.
/// 2. If a dimension has `size == 1` and `offset != 0`, it can be folded into
/// another dimension with the same stride if that exists.
///
/// Example for case 1:
///
/// offsets: [0, 0, 0], sizes: [32, 1, 8], strides: [32, 1024, 1]
///
/// will be transformed into:
///
/// offsets: [0, 0], sizes: [32, 8], strides: [32, 1]
///
/// Example for case 2:
///
/// offsets: [1, 0, 1, 0], sizes: [1, 32, 1, 8], strides: [1024, 32, 1024, 1]
///
/// will be transformed into:
///
/// offsets: [2, 0, 0], sizes: [1, 32, 8], strides: [1024, 32, 1]
///
/// Note that the dimensions are merged into the outermost one. Heuristically,
/// this works out best with other strided access pattern transformations, but
/// could be made an option in the future.
LogicalResult foldUnitDims(MLIRContext *ctx,
                           const SmallVector<OpFoldResult> &offsets,
                           const SmallVector<OpFoldResult> &strides,
                           const SmallVector<OpFoldResult> &sizes,
                           SmallVector<OpFoldResult> &newOffsets,
                           SmallVector<OpFoldResult> &newStrides,
                           SmallVector<OpFoldResult> &newSizes);

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

  DmaDimConfig(const AMDAIE::AMDAIEDeviceModel &deviceModel, uint8_t memSpace)
      : deviceModel(deviceModel) {
    if (memSpace == 0) {
      uint8_t shimNbIntraDims = deviceModel.getDmaProp<uint8_t>(
          AMDAIE::AMDAIETileType::SHIMNOC, AMDAIE::AMDAIEDmaProp::NumAddrDim);
      tileType = AMDAIE::AMDAIETileType::SHIMNOC;
      nbInterDims = deviceModel.deviceConfig.dmaNbInterDims;
      maxNbDims = shimNbIntraDims + nbInterDims;
    } else if (memSpace == 1) {
      uint8_t memTileNbIntraDims = deviceModel.getDmaProp<uint8_t>(
          AMDAIE::AMDAIETileType::MEMTILE, AMDAIE::AMDAIEDmaProp::NumAddrDim);
      tileType = AMDAIE::AMDAIETileType::MEMTILE;
      maxNbDims = memTileNbIntraDims;
    } else if (memSpace == 2) {
      uint8_t coreNbIntraDims = deviceModel.getDmaProp<uint8_t>(
          AMDAIE::AMDAIETileType::AIETILE, AMDAIE::AMDAIEDmaProp::NumAddrDim);
      tileType = AMDAIE::AMDAIETileType::AIETILE;
      maxNbDims = coreNbIntraDims;
    } else {
      assert(false && "unsupported memspace: ");
    }
  }

  /// Return a vector containing the max size values for every dimension.
  SmallVector<int64_t> getMaxSizes() const;

  /// Return a vector containing the max stride values for every dimension.
  SmallVector<int64_t> getMaxStrides() const;
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
