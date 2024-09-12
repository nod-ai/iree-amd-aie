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

// Constant specifying the number of inter-iteration dimension for DMA
// operations.
//
// NOTE(jornt): this number is implicitly assumed in the device model and can't
// be retrieved from it afaik.
//
// Some background:
//
// DMAs support multi-dimensional addressing through buffer descriptors in two
// ways:
// 1. Intra-iteration access pattern. Specified via 'strides' ('steps' in buffer
// descriptor lingo), 'sizes' ('wraps' in buffer descriptro lingo) and
// 'padding'. When a DMA executes a buffer descriptor, it will access the data
// (read/write) as specified by the intra-iteration access pattern.
// 2. Inter-iteration access pattern. Specified via an iteration 'stride',
// 'size' and 'current_iteration' ('stride' is the same as 'stepsize' and 'size'
// is the same as 'wrap' in buffer descriptor lingo). Here, 'current_iteration'
// keeps track of the current execution iteration of the buffer descriptor and
// is incremented after buffer descriptor execution. the 'stride' is the offset
// to be used for each execution of the buffer descriptor, relative to the
// previous one. When 'iteration_current' is equal to 'size', the
// 'iteration_current' is reset to zero.
//
// Although DMAs can have a different number of intra-iteration dimensions, all
// DMAs have a single inter-iteration dimension (at least in AIE2 and AIE2p).
static const size_t kAMDAIEDmaNbInterDims = 1;

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
LogicalResult foldLinearDims(MLIRContext *ctx,
                             const SmallVector<OpFoldResult> &offsets,
                             const SmallVector<OpFoldResult> &sizes,
                             const SmallVector<OpFoldResult> &strides,
                             SmallVector<OpFoldResult> &newOffsets,
                             SmallVector<OpFoldResult> &newSizes,
                             SmallVector<OpFoldResult> &newStrides);

/// Fold single dimension linear accesses and make them implicit. `This
/// operation happens in place. Returns `success` if folding took place.
LogicalResult foldSingleDim(SmallVector<OpFoldResult> &offsets,
                            SmallVector<OpFoldResult> &sizes,
                            SmallVector<OpFoldResult> &strides);

/// Fold unit dimensions within a strided access pattern. Returns `success` if
/// folding took place.
LogicalResult foldUnitDims(const SmallVector<OpFoldResult> &offsets,
                           const SmallVector<OpFoldResult> &strides,
                           const SmallVector<OpFoldResult> &sizes,
                           SmallVector<OpFoldResult> &newOffsets,
                           SmallVector<OpFoldResult> &newStrides,
                           SmallVector<OpFoldResult> &newSizes);

/// Utility to discard all non-zero offsets that have dimension equal to 1 on
/// the same index of the provided shape. This helps with updating DMA
/// operations for a shape change. If an empty shape is passed, all non-zero
/// offsets will be removed.
template <CopyOpOperateOn OperateOn>
AMDAIE::DoublyStridedOpInterface discardAllNonZeroOffsets(
    RewriterBase &rewriter, AMDAIE::DoublyStridedOpInterface op,
    SmallVector<int64_t> &shape) {
  SmallVector<OpFoldResult> newSourceOffsets;
  SmallVector<OpFoldResult> newSourceSizes;
  SmallVector<OpFoldResult> newSourceStrides;
  SmallVector<OpFoldResult> newTargetOffsets;
  SmallVector<OpFoldResult> newTargetSizes;
  SmallVector<OpFoldResult> newTargetStrides;
  if constexpr (OperateOn == CopyOpOperateOn::Source) {
    SmallVector<OpFoldResult> offsets = op.getSourceMixedOffsets();
    SmallVector<OpFoldResult> sizes = op.getSourceMixedSizes();
    SmallVector<OpFoldResult> strides = op.getSourceMixedStrides();
    // Set shape to a vector of ones as a default.
    if (shape.empty()) {
      SmallVector<int64_t> ones(offsets.size(), 1);
      shape = ones;
    }
    if (shape.size() != offsets.size()) return op;
    // Fill source offsets/sizes/strides.
    for (auto &&[offset, size, stride, dim] :
         llvm::zip(offsets, sizes, strides, shape)) {
      std::optional<int64_t> constantOffset = getConstantIntValue(offset);
      if (dim == 1 && !constantOffset) continue;
      if (dim == 1 && constantOffset && constantOffset.value() != 0) continue;
      newSourceOffsets.push_back(offset);
      newSourceSizes.push_back(size);
      newSourceStrides.push_back(stride);
    }
    newTargetOffsets = op.getTargetMixedOffsets();
    newTargetSizes = op.getTargetMixedSizes();
    newTargetStrides = op.getTargetMixedStrides();
  } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
    SmallVector<OpFoldResult> offsets = op.getTargetMixedOffsets();
    SmallVector<OpFoldResult> sizes = op.getTargetMixedSizes();
    SmallVector<OpFoldResult> strides = op.getTargetMixedStrides();
    // Set shape to a vector of ones as a default.
    if (shape.empty()) {
      SmallVector<int64_t> ones(offsets.size(), 1);
      shape = ones;
    }
    if (shape.size() != offsets.size()) return op;
    // Fill source offsets/sizes/strides.
    for (auto &&[offset, size, stride, dim] :
         llvm::zip(offsets, sizes, strides, shape)) {
      std::optional<int64_t> constantOffset = getConstantIntValue(offset);
      if (dim == 1 && !constantOffset) continue;
      if (dim == 1 && constantOffset && constantOffset.value() != 0) continue;
      newTargetOffsets.push_back(offset);
      newTargetSizes.push_back(size);
      newTargetStrides.push_back(stride);
    }
    newSourceOffsets = op.getSourceMixedOffsets();
    newSourceSizes = op.getSourceMixedSizes();
    newSourceStrides = op.getSourceMixedStrides();
  }
  rewriter.setInsertionPointAfter(op);
  auto newDoublyStridedOp = op.createDoublyStridedOp(
      rewriter, newTargetOffsets, newTargetSizes, newTargetStrides,
      newSourceOffsets, newSourceSizes, newSourceStrides);
  rewriter.replaceOp(op, newDoublyStridedOp.getOperation());
  return newDoublyStridedOp;
}

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
  AMDAIE::AMDAIETileType sourceTileType;
  AMDAIE::AMDAIETileType targetTileType;
  /// The maximum number of addressing dimensions on the source side of the DMA.
  uint8_t sourceMaxNbDims{0};
  /// The maximum number of addressing dimensions on the target side of the DMA.
  uint8_t targetMaxNbDims{0};

  DmaDimConfig(const AMDAIE::AMDAIEDeviceModel &deviceModel,
               uint8_t sourceMemspaceInt, uint8_t targetMemspaceInt)
      : deviceModel(deviceModel) {
    uint8_t shimNbIntraDims = deviceModel.getDmaProp<uint8_t>(
        AMDAIE::AMDAIETileType::SHIMNOC, AMDAIE::AMDAIEDmaProp::NumAddrDim);
    uint8_t memTileNbIntraDims = deviceModel.getDmaProp<uint8_t>(
        AMDAIE::AMDAIETileType::MEMTILE, AMDAIE::AMDAIEDmaProp::NumAddrDim);
    uint8_t coreNbIntraDims = deviceModel.getDmaProp<uint8_t>(
        AMDAIE::AMDAIETileType::AIETILE, AMDAIE::AMDAIEDmaProp::NumAddrDim);
    if (sourceMemspaceInt == 0) {
      sourceTileType = AMDAIE::AMDAIETileType::SHIMNOC;
      sourceMaxNbDims = shimNbIntraDims + kAMDAIEDmaNbInterDims;
    } else if (sourceMemspaceInt == 1) {
      sourceTileType = AMDAIE::AMDAIETileType::MEMTILE;
      sourceMaxNbDims = memTileNbIntraDims;
    } else if (sourceMemspaceInt == 2) {
      sourceTileType = AMDAIE::AMDAIETileType::AIETILE;
      sourceMaxNbDims = coreNbIntraDims;
    } else {
      assert(false && "unsupported source memspace");
    }
    if (targetMemspaceInt == 0) {
      targetTileType = AMDAIE::AMDAIETileType::SHIMNOC;
      targetMaxNbDims = shimNbIntraDims + kAMDAIEDmaNbInterDims;
    } else if (targetMemspaceInt == 1) {
      targetTileType = AMDAIE::AMDAIETileType::MEMTILE;
      targetMaxNbDims = memTileNbIntraDims;
    } else if (targetMemspaceInt == 2) {
      targetTileType = AMDAIE::AMDAIETileType::AIETILE;
      targetMaxNbDims = coreNbIntraDims;
    } else {
      assert(false && "unsupported target memspace");
    }
  }

  /// Return a vector containing the max stride values for every dimension. The
  /// first dimension is the inter-iteration dimension, while the latter are
  /// intra-iteration dimensions.
  /// NOTE: It doesn't need to be known which BDs will be used exactly as all
  /// BDs on the same tile types should have the same step and wrap sizes.
  /// Therefore, `BD ID == 0` is choosen to be used to retrieve device
  /// information.
  template <CopyOpOperateOn OperateOn>
  SmallVector<uint32_t> getMaxStrides() const {
    uint32_t maxIntraStride;
    uint32_t maxInterStride;
    if constexpr (OperateOn == CopyOpOperateOn::Source) {
      maxIntraStride = deviceModel.getDmaBdProp<uint32_t>(
          sourceTileType, 0, AMDAIE::AMDAIEDmaBdProp::StepSizeMax);
      maxInterStride = deviceModel.getDmaBdProp<uint32_t>(
          sourceTileType, 0, AMDAIE::AMDAIEDmaBdProp::IterStepSizeMax);
      // +1 because values are encoded in HW BDs as (value - 1), so the range is
      // [1:2^x].
      SmallVector<uint32_t> stepSizes(sourceMaxNbDims, maxIntraStride + 1);
      stepSizes[0] = maxInterStride + 1;
      return stepSizes;
    } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
      maxIntraStride = deviceModel.getDmaBdProp<uint32_t>(
          targetTileType, 0, AMDAIE::AMDAIEDmaBdProp::StepSizeMax);
      maxInterStride = deviceModel.getDmaBdProp<uint32_t>(
          targetTileType, 0, AMDAIE::AMDAIEDmaBdProp::IterStepSizeMax);
      // +1 because values are encoded in HW BDs as (value - 1), so the range is
      // [1:2^x].
      SmallVector<uint32_t> stepSizes(targetMaxNbDims, maxIntraStride + 1);
      stepSizes[0] = maxInterStride + 1;
      return stepSizes;
    } else {
      assert(false && "Function can only operate on Source or Target");
    }
  }

  /// Return a vector containing the max size values for every dimension. The
  /// first dimension is the inter-iteration dimension, while the latter are
  /// intra-iteration dimensions.
  /// NOTE: It doesn't need to be known which BDs will be used exactly as all
  /// BDs on the same tile types should have the same step and wrap sizes.
  /// Therefore, `BD ID == 0` is choosen to be used to retrieve device
  /// information.
  template <CopyOpOperateOn OperateOn>
  SmallVector<uint32_t> getMaxSizes() const {
    uint32_t maxIntraSize;
    uint32_t maxInterSize;
    if constexpr (OperateOn == CopyOpOperateOn::Source) {
      maxIntraSize = deviceModel.getDmaBdProp<uint16_t>(
          sourceTileType, 0, AMDAIE::AMDAIEDmaBdProp::WrapMax);
      maxInterSize = deviceModel.getDmaBdProp<uint8_t>(
          sourceTileType, 0, AMDAIE::AMDAIEDmaBdProp::IterWrapMax);
      SmallVector<uint32_t> stepSizes(sourceMaxNbDims, maxIntraSize);
      stepSizes[0] = maxInterSize;
      return stepSizes;
    } else if constexpr (OperateOn == CopyOpOperateOn::Target) {
      maxIntraSize = deviceModel.getDmaBdProp<uint16_t>(
          targetTileType, 0, AMDAIE::AMDAIEDmaBdProp::WrapMax);
      maxInterSize = deviceModel.getDmaBdProp<uint8_t>(
          targetTileType, 0, AMDAIE::AMDAIEDmaBdProp::IterWrapMax);
      SmallVector<uint32_t> stepSizes(targetMaxNbDims, maxIntraSize);
      stepSizes[0] = maxInterSize;
      return stepSizes;
    } else {
      assert(false && "Function can only operate on Source or Target");
    }
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
