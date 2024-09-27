// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-amdaie-convert-to-dma"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Applies packing to a given input.
LogicalResult updateFromPack(IREE::LinalgExt::PackOp packOp,
                             SmallVector<OpFoldResult> &offsets,
                             SmallVector<OpFoldResult> &sizes,
                             SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = packOp.getContext();

  llvm::ArrayRef<int64_t> permutation = packOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = packOp.getStaticInnerTiles();
  ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();

  assert(offsets.size() == sizes.size() && sizes.size() == strides.size() &&
         "offsets, sizes, and strides must have the same size.");
  for (int64_t dim : innerDimsPos) {
    assert(dim < sizes.size() && "innerDimsPos must be within sizes.");
  }

  SmallVector<OpFoldResult> innerSizes;
  SmallVector<OpFoldResult> innerStrides;
  SmallVector<OpFoldResult> innerOffsets;

  for (int i = 0; i < innerTiles.size(); i++) {
    // Calculate new sizes.
    innerSizes.push_back(getAsIndexOpFoldResult(ctx, innerTiles[i]));
    std::optional<int64_t> maybeSize =
        getConstantIntValue(sizes[innerDimsPos[i]]);
    if (!maybeSize.has_value()) {
      packOp->emitOpError("requires all constant sizes.");
    }
    int64_t size = maybeSize.value();
    if (size % innerTiles[i] != 0) {
      auto message = llvm::formatv(
          "in dimension {0}, the tile size {1} does not divide the tensor size "
          "{2}. Imperfect/partial tiling is currently not supported.",
          i, innerTiles[i], size);
      return packOp->emitOpError(message);
    }
    sizes[innerDimsPos[i]] = getAsIndexOpFoldResult(ctx, size / innerTiles[i]);

    // The tiled dim inherits the stride from the corresponding outer dim and
    // the outer dims stride gets multiplied by the size of the tile.
    innerStrides.push_back(strides[innerDimsPos[i]]);
    std::optional<int64_t> maybeStride =
        getConstantIntValue(strides[innerDimsPos[i]]);
    if (!maybeStride.has_value())
      packOp->emitOpError("requires a constant stride here.");
    int64_t stride = maybeStride.value();
    strides[innerDimsPos[i]] =
        getAsIndexOpFoldResult(ctx, stride * innerTiles[i]);

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
LogicalResult updateFromUnPack(IREE::LinalgExt::UnPackOp unPackOp,
                               SmallVector<OpFoldResult> &offsets,
                               SmallVector<OpFoldResult> &sizes,
                               SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = unPackOp.getContext();

  llvm::ArrayRef<int64_t> permutation = unPackOp.getOuterDimsPerm();
  llvm::ArrayRef<int64_t> innerTiles = unPackOp.getStaticInnerTiles();
  llvm::ArrayRef<int64_t> innerDimsPos = unPackOp.getInnerDimsPos();

  SmallVector<OpFoldResult> innerSizes;
  SmallVector<OpFoldResult> innerStrides;
  SmallVector<OpFoldResult> innerOffsets;

  int nbOuterDims = sizes.size() - innerTiles.size();
  SmallVector<OpFoldResult> outerOffsets{offsets.begin(),
                                         offsets.begin() + nbOuterDims};
  SmallVector<OpFoldResult> outerStrides{strides.begin(),
                                         strides.begin() + nbOuterDims};
  SmallVector<OpFoldResult> outerSizes{sizes.begin(),
                                       sizes.begin() + nbOuterDims};

  // Apply inverse permutation to the outer dims if permutation provided (if
  // permutation not provided, it is identity, and therefore so is the inverse).
  if (!permutation.empty()) {
    SmallVector<int64_t> inversePermutation =
        invertPermutationVector(permutation);
    applyPermutationToVector(outerStrides, inversePermutation);
    applyPermutationToVector(outerSizes, inversePermutation);
    applyPermutationToVector(outerOffsets, inversePermutation);
  }

  // Do the unpacking on the outer dims.
  llvm::SmallVector<int64_t> outerDimsIndexMap(nbOuterDims, 0);
  // Intialize the indexing of each outer dim.
  std::iota(outerDimsIndexMap.begin(), outerDimsIndexMap.end(), 0);

  for (uint64_t i = 0; i < innerTiles.size(); i++) {
    int64_t insertionIndex = outerDimsIndexMap[innerDimsPos[i]] + 1;
    // Insert inner dims adjacent to their corresponding outer dims.
    outerSizes.insert(outerSizes.begin() + insertionIndex,
                      getAsIndexOpFoldResult(ctx, innerTiles[i]));
    outerStrides.insert(outerStrides.begin() + insertionIndex,
                        strides[nbOuterDims + i]);
    outerOffsets.insert(outerOffsets.begin() + insertionIndex,
                        offsets[nbOuterDims + i]);
    // Update the map as all the dimensions inner to the innerDimsPos[i] are now
    // shifted by 1.
    for (uint64_t j = innerDimsPos[i] + 1; j < nbOuterDims; j++) {
      outerDimsIndexMap[j]++;
    }
  }
  // Make the outer dims as the final returned dims
  offsets = outerOffsets;
  strides = outerStrides;
  sizes = outerSizes;
  return success();
}

static bool isAllocation(Operation *op) {
  return op && (isa<memref::AllocOp>(op) ||
                isa<IREE::HAL::InterfaceBindingSubspanOp>(op));
}

/// Initialize offsets, sizes, and strides from an allocation operation.
LogicalResult setFromAlloc(Operation *op, SmallVector<OpFoldResult> &offsets,
                           SmallVector<OpFoldResult> &sizes,
                           SmallVector<OpFoldResult> &strides) {
  assert(isAllocation(op) &&
         "expected memref.alloc or hal.interface.binding.subspan.");

  MemRefType memRefType = cast<MemRefType>(op->getResult(0).getType());
  MLIRContext *ctx = memRefType.getContext();
  auto [stridesI64, baseOffset] = getStridesAndOffset(memRefType);
  strides = getAsIndexOpFoldResult(ctx, stridesI64);

  if (baseOffset != 0) {
    auto message = llvm::formatv(
        "has non-zero offset {0} which is currently unsupported.", baseOffset);
    return op->emitOpError(message);
  }
  offsets.resize(strides.size(), getAsIndexOpFoldResult(ctx, 0));

  ArrayRef<int64_t> sizesI64 = memRefType.getShape();
  if (llvm::any_of(sizesI64,
                   [](int64_t size) { return ShapedType::isDynamic(size); })) {
    return op->emitOpError("has dynamic size, which is unsupported in DMA.");
  }
  sizes = getAsIndexOpFoldResult(ctx, sizesI64);
  return success();
}

/// Return `lhs` + `rhs`, where `lhs` and `rhs` are OpFoldResults of integers.
///
/// The implementation considers the 4 cases of 
///   (`lhs`, `rhs`) in {Attribute, Value} x {Attribute, Value}.
SmallVector<OpFoldResult> getIndexOpFoldResultSum(OpBuilder &builder,
                                                  Location loc,
                                                  ArrayRef<OpFoldResult> lhs,
                                                  ArrayRef<OpFoldResult> rhs) {
  assert(lhs.size() == rhs.size() && "lhs and rhs not same size.");
  SmallVector<OpFoldResult> sum;
  sum.reserve(lhs.size());

  auto getConstant = [&](int64_t v) -> Value {
    return builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(),
        builder.getIntegerAttr(builder.getIndexType(), v));
  };

  auto add = [&](Value v, IntegerAttr attr) {
    if (attr.getInt() == 0) return v;
    return builder.create<arith::AddIOp>(loc, v, getConstant(attr.getInt()))
        .getResult();
  };

  for (uint64_t i = 0; i < lhs.size(); ++i) {
    IntegerAttr aAttr;
    if (auto aAttr_ = dyn_cast<Attribute>(lhs[i])) {
      aAttr = dyn_cast<IntegerAttr>(aAttr_);
      assert(aAttr && "Expected IntegerAttr.");
    }

    IntegerAttr bAttr;
    if (auto bAttr_ = dyn_cast<Attribute>(rhs[i])) {
      bAttr = dyn_cast<IntegerAttr>(bAttr_);
      assert(bAttr && "Expected IntegerAttr.");
    }

    if (aAttr && bAttr) {
      sum.push_back(getAsIndexOpFoldResult(builder.getContext(),
                                           aAttr.getInt() + bAttr.getInt()));
    } else if (!aAttr && !bAttr) {
      sum.push_back(builder
                        .create<arith::AddIOp>(loc, cast<Value>(lhs[i]),
                                               cast<Value>(rhs[i]))
                        .getResult());
    } else if (!aAttr && bAttr) {
      sum.push_back(add(cast<Value>(lhs[i]), bAttr));
    } else if (aAttr && !bAttr) {
      sum.push_back(add(cast<Value>(rhs[i]), aAttr));
    } else {
      assert(false && "unreachable");
    }
  }

  return sum;
}

/// Return sum_{i} values[i] * coeffs[i], where 
///
/// - values are OpFoldResults (i.e. each element in `values` is 
///                                  either an mlir::Value or mlir::Attribute)
/// - coeffs are integers.
OpFoldResult getLinearCombination(OpBuilder &builder, Location loc,
                                  ArrayRef<OpFoldResult> values,
                                  ArrayRef<int64_t> coeffs) {
  assert(values.size() == coeffs.size() && "values and coeffs not same size");
  MLIRContext *ctx = builder.getContext();

  auto getConstant = [&](int64_t v) -> Value {
    return builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(),
        builder.getIntegerAttr(builder.getIndexType(), v));
  };

  // Initialize the linear combination to 0.
  OpFoldResult lc = builder.getIndexAttr(0);

  // For eacho of the (value, coeff) pairs, add the product to the linear
  // combination, updating `lc` in each iteration. The implementation
  // here is careful not to create constant zero values.
  for (uint64_t dim = 0; dim < coeffs.size(); ++dim) {
    // Four cases considered:
    // 1) `values[dim]` is an attribute (constant) and `lc` is also
    //     an attribute (constant)
    // 2) `values[dim]` is an attribute (constant) and `lc` is a Value
    //     (non-constant)
    // 3) `values[dim]` is a Value (non-constant) and `lc` is an
    //     attribute (constant)
    // 4) `values[dim]` is a Value (non-constant) and `lc` is also a
    //     Value (non-constant)
    if (auto valueAttr = dyn_cast<Attribute>(values[dim])) {
      int64_t term = coeffs[dim] * cast<IntegerAttr>(valueAttr).getInt();
      // Case 1.
      if (auto lcAttr = dyn_cast<Attribute>(lc)) {
        lc = getAsIndexOpFoldResult(ctx,
                                    term + cast<IntegerAttr>(lcAttr).getInt());
      }
      // Case 2.
      else if (term != 0) {
        lc = builder
                 .create<arith::AddIOp>(loc, cast<Value>(lc), getConstant(term))
                 .getResult();
      }
    } else {
      Value term = builder.create<arith::MulIOp>(loc, cast<Value>(values[dim]),
                                                 getConstant(coeffs[dim]));
      // Case 3.
      if (auto lcAttr = dyn_cast<Attribute>(lc)) {
        int64_t c = cast<IntegerAttr>(lcAttr).getInt();
        if (c != 0) {
          lc = builder.create<arith::AddIOp>(loc, getConstant(c), term)
                   .getResult();
        } else {
          lc = term;
        }
      }
      // Case 4.
      else {
        lc = builder.create<arith::AddIOp>(loc, cast<Value>(lc), term)
                 .getResult();
      }
    }
  }
  return lc;
}

/// Update the offsets, sizes, and strides from a collapse shape operation.
LogicalResult updateFromCollapseShape(memref::CollapseShapeOp collapseOp,
                                      SmallVector<OpFoldResult> &offsets,
                                      SmallVector<OpFoldResult> &sizes,
                                      SmallVector<OpFoldResult> &strides) {
  auto reassociationIndices = collapseOp.getReassociationIndices();
  ArrayRef<int64_t> inputShape = collapseOp.getSrcType().getShape();
  ArrayRef<int64_t> resultShape = collapseOp.getType().getShape();
  uint64_t resultRank = resultShape.size();
  MLIRContext *ctx = collapseOp.getContext();

  // Set strides to inner-most stride in each reassocation group.
  //
  // Example: Consider a 2x3x5x7 tensor, with strides [70,35,7,1]. If this 
  // is collapsed to shape 6x35, the srides are [35,1]. The reassociation
  // groups are [0,1] and [2,3], and so we've just taken the inner-most
  // strides in each group.
  for (auto reassociation : llvm::enumerate(reassociationIndices)) {
    uint64_t index = reassociation.index();
    uint64_t dim = reassociation.value().back();
    strides[index] = strides[dim];
  }
  strides.resize(resultRank);

  // Set sizes to output shape, and check that all dims are static.
  sizes.clear();
  for (int64_t dim : resultShape) {
    if (dim == ShapedType::kDynamic) {
      return collapseOp.emitOpError(
          "has a dynamic shape which is currently unsupported.");
    }
    sizes.push_back(getAsIndexOpFoldResult(ctx, dim));
  }

  // Offsets - merge reassocation groups by taking linear combinations of the
  // offsets with local strides. Using the example of the shape of 2x3x5x7
  // being collapsed to 6x35, if the initial offsets are [a,b,c,d], the 
  // collapsed offsets are [a*3 + b, c*7 + d].
  SmallVector<OpFoldResult> collapsedOffsets;
  for (auto reassociation : llvm::enumerate(reassociationIndices)) {
    auto dims = reassociation.value();

    // The strides within the group:
    SmallVector<int64_t> localStrides(dims.size(), 1);
    for (uint64_t i = 1; i < dims.size(); ++i) {
      uint64_t dim = dims.size() - i - 1;
      localStrides[dim] = localStrides[dim + 1] * inputShape[dims[dim + 1]];
    }

    OpBuilder builder(ctx);
    builder.setInsertionPoint(collapseOp);
    OpFoldResult combination = getLinearCombination(
        builder, collapseOp.getLoc(),
        ArrayRef<OpFoldResult>(offsets.begin() + dims[0],
                               offsets.begin() + dims.back() + 1),
        localStrides);
    collapsedOffsets.push_back(combination);
  }
  offsets = collapsedOffsets;
  assert(offsets.size() == sizes.size() && sizes.size() == strides.size() &&
         "mismatch in the number of offsets, sizes and strides");
  return success();
}

/// Update the offsets, sizes, and strides from an expand shape operation.
LogicalResult updateFromExpandShape(memref::ExpandShapeOp expandShapeOp,
                                    SmallVector<OpFoldResult> &offsets,
                                    SmallVector<OpFoldResult> &sizes,
                                    SmallVector<OpFoldResult> &strides) {
  MLIRContext *ctx = expandShapeOp.getContext();
  auto reassociationIndices = expandShapeOp.getReassociationIndices();
  ArrayRef<int64_t> resultShape = expandShapeOp.getType().getShape();

  // Set the sizes to the output shape, and check that all dims are static.
  SmallVector<OpFoldResult> newSizes(resultShape.size());
  for (int i = 0; i < resultShape.size(); i++) {
    if (resultShape[i] == ShapedType::kDynamic) {
      return expandShapeOp.emitOpError(
          "has a dynamic shape which is currently unsupported.");
    }
    newSizes[i] = getAsIndexOpFoldResult(ctx, resultShape[i]);
  }

  // Strides. Using the example expanding from a shape of 6x35 to 2x3x5x7, where
  // the initial strides are [50, 1], the new strides will be [150, 50, 7, 1].
  SmallVector<OpFoldResult> newStrides(resultShape.size());
  for (auto reassociation : llvm::enumerate(reassociationIndices)) {
    uint64_t index = reassociation.index();
    auto dims = reassociation.value();
    OpFoldResult stride = strides[index];
    if (!isa<Attribute>(stride)) {
      return expandShapeOp.emitOpError("cannot operate on a dynamic stride.");
    }
    int64_t cum = getConstantIntValue(stride).value();
    for (uint64_t i = 0; i < dims.size(); i++) {
      uint64_t d = dims[dims.size() - i - 1];
      newStrides[d] = getAsIndexOpFoldResult(ctx, cum);
      cum *= resultShape[d];
    }
  }

  // Offsets. For now we don't do any arithmetic to split the offset across
  // dimensions, in theory we need to split the offset amongst the reassociation
  // indices, but for now we're just putting the offset on the inner most
  // dimension.
  //
  // Example: suppose we're expanding from 6x35 to 2x3x5x7, and the initial
  // offsets are [a, b]. The new offsets will be [0, a, 0, b]. In theory they 
  // should be [a/3, a%3, b/7 b%7] but these offsets ultimately get collapsed 
  // anyway so it doesn't matter if we don't. 
  SmallVector<OpFoldResult> newOffsets(resultShape.size());
  // Initialize all ofsets to 0:
  for (int i = 0; i < resultShape.size(); i++) {
    newOffsets[i] = getAsIndexOpFoldResult(ctx, 0);
  }
  // Populate the inner-most dimensions with the original offsets:
  for (auto reassociation : llvm::enumerate(reassociationIndices)) {
    uint64_t index = reassociation.index();
    auto dims = reassociation.value();
    newOffsets[dims.back()] = offsets[index];
  }

  sizes = newSizes;
  offsets = newOffsets;
  strides = newStrides;
  return success();
}


/// Update the offsets, sizes, and strides from a subview operation.
LogicalResult updateFromSubView(memref::SubViewOp subviewOp,
                                SmallVector<OpFoldResult> &offsets,
                                SmallVector<OpFoldResult> &sizes,
                                SmallVector<OpFoldResult> &strides) {
  assert(offsets.size() == subviewOp.getMixedSizes().size());

  OpBuilder builder(subviewOp.getContext());
  builder.setInsertionPoint(subviewOp);
  offsets = getIndexOpFoldResultSum(builder, subviewOp.getLoc(), offsets,
                                    subviewOp.getMixedOffsets());

  sizes = subviewOp.getMixedSizes();
  if (llvm::any_of(sizes, [](OpFoldResult size) {
        return !getConstantIntValue(size).has_value();
      })) {
    return subviewOp->emitOpError(
        "has dynamic shape that is not supported by the target dma op.");
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
    } else {
      // TODO(newling) add a test of this path. 
      // If the offset is non-zero, we shouldn't just be dropping it. For now,
      // just bail.
      OpFoldResult offset = offsets[extractionIndex];
      if (isa<Value>(offset) || getConstantIntValue(offset).value() != 0) {
        return subviewOp->emitOpError(
            "cannot update a non-zero offset in a dimension that is being "
            "dropped.");
      }
    }
  }
  offsets.resize(insertionIndex);
  sizes.resize(insertionIndex);
  strides.resize(insertionIndex);
  return success();
}

/// Provide the offsets, sizes and strides of the inputs to `operandOp`.
/// This function updates `operandOp`, setting it to the allocation operation
/// that it originates from.
LogicalResult setDmaInputs(Operation *&operandOp,
                           SmallVector<OpFoldResult> &offsets,
                           SmallVector<OpFoldResult> &sizes,
                           SmallVector<OpFoldResult> &strides) {
  assert(offsets.empty() && sizes.empty() && strides.empty() &&
         "offsets, sizes, and strides must be empty");

  if (!operandOp) assert(false && "operandOp must be non-null");

  // Get the sequence of memref operations going from an allocation to
  // `operandOp`
  SmallVector<Operation *> chain;

  Operation *currentOp = operandOp;
  while (currentOp) {
    chain.push_back(currentOp);
    if (isAllocation(currentOp)) {
      currentOp = {};
    } else if (auto memref = dyn_cast<memref::SubViewOp>(currentOp)) {
      currentOp = memref.getSource().getDefiningOp();
    } else if (auto expand = dyn_cast<memref::ExpandShapeOp>(currentOp)) {
      currentOp = expand.getSrc().getDefiningOp();
    } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(currentOp)) {
      currentOp = collapse.getSrc().getDefiningOp();
    } else {
      return currentOp->emitOpError("is currently not supported");
    }
  }

  operandOp = chain.back();
  if (!isAllocation(operandOp)) {
    return operandOp->emitOpError(
        "is not a memref.alloc or hal.interface.binding.subspan operation.");
  }

  // Starting from the allocation, update the offsets, sizes, and strides.
  for (auto iter = chain.rbegin(); iter != chain.rend(); ++iter) {
    Operation *op = *iter;
    if (isAllocation(op)) {
      if (failed(setFromAlloc(op, offsets, sizes, strides))) {
        return failure();
      }
    } else if (auto memref = dyn_cast<memref::SubViewOp>(op)) {
      if (failed(updateFromSubView(memref, offsets, sizes, strides))) {
        return failure();
      }
    } else if (auto expand = dyn_cast<memref::ExpandShapeOp>(op)) {
      if (failed(updateFromExpandShape(expand, offsets, sizes, strides))) {
        return failure();
      }
    } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(op)) {
      if (failed(updateFromCollapseShape(collapse, offsets, sizes, strides))) {
        return failure();
      }
    } else {
      return op->emitOpError("is currently not supported");
    }
  }
  return success();
}

/// Get the inputs from the pack/unpack op 'op'. Return failure if 'op' is not
/// a pack/unpack op, or if 'op' is determined unlowerable to a DMA operation.
LogicalResult processInputs(Operation *packOrUnackOp,
                            SmallVector<OpFoldResult> &offsets,
                            SmallVector<OpFoldResult> &sizes,
                            SmallVector<OpFoldResult> &strides) {
  if (auto packOp = dyn_cast<IREE::LinalgExt::PackOp>(packOrUnackOp)) {
    if (failed(updateFromPack(packOp, offsets, sizes, strides))) {
      return failure();
    }
  } else if (auto unPackOp =
                 dyn_cast<IREE::LinalgExt::UnPackOp>(packOrUnackOp)) {
    if (failed(updateFromUnPack(unPackOp, offsets, sizes, strides))) {
      return failure();
    }
  } else {
    assert(false && "expected pack/unpack op in processInputs");
  }
  return success();
}

/// Rewrite the pack/unpack op 'op' as a DMA operation. The function arguments
/// 'input', 'output', and 'innerTiles' are the input, output, and inner tile
/// of 'op'. If 'op' is determined to not currently be lowerable to a DMA
/// operation, failure is returned.
///
/// Design note: arguments 'input', 'output', and 'innerTiles' could be
/// obtained from 'op' inside this function if it were templatized, but
/// I've factorized out that logic to reduce the total amount of templatized
/// code.
LogicalResult rewriteAsDma(IRRewriter &rewriter, Operation *packOrUnackOp,
                           Value input, Value output,
                           llvm::ArrayRef<int64_t> innerTiles) {
  assert(packOrUnackOp && "packOrUnackOp is null");

  if (llvm::any_of(innerTiles,
                   [](int64_t size) { return ShapedType::isDynamic(size); })) {
    packOrUnackOp->emitError(
        "has a non-static shape: not yet supported by this pass.");
  }

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(packOrUnackOp);

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

  if (!succeeded(
          processInputs(packOrUnackOp, srcOffsets, srcShape, srcBaseStrides))) {
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

  rewriter.setInsertionPoint(packOrUnackOp);
  rewriter.create<AMDAIE::DmaCpyNdOp>(packOrUnackOp->getLoc(), dst, dstOffsets,
                                      dstShape, dstBaseStrides, src, srcOffsets,
                                      srcShape, srcBaseStrides);

  rewriter.eraseOp(packOrUnackOp);
  return success();
}

template <typename PackOrUnpackOp>
LogicalResult rewriteAsDma(PackOrUnpackOp op, IRRewriter &rewriter) {
  Value input = op.getInput();
  Value output = op.getOutput();
  llvm::ArrayRef<int64_t> innerTiles = op.getStaticInnerTiles();
  return rewriteAsDma(rewriter, op, input, output, innerTiles);
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
      getOperation()->walk([&rewriter](linalg::CopyOp copyOp) {
        if (failed(copyToPack(rewriter, copyOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  if (convertCopiesWalkResult.wasInterrupted()) return signalPassFailure();

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

std::unique_ptr<Pass> createAMDAIEConvertToDmaPass() {
  return std::make_unique<AMDAIEConvertToDmaPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
