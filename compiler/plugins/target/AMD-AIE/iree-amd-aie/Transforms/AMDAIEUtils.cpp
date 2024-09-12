// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIEUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Iterators.h"

namespace mlir::iree_compiler::AMDAIE {

std::optional<AMDAIEDevice> getConfigAMDAIEDevice(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (!targetAttr) return std::nullopt;
  auto config = targetAttr.getConfiguration();
  if (!config) return std::nullopt;
  std::optional<StringAttr> attr = config.getAs<StringAttr>("target_device");
  if (!attr) return std::nullopt;
  return AMDAIE::symbolizeEnum<AMDAIEDevice>(attr.value().getValue());
}

std::optional<AMDAIEDevice> getConfigAMDAIEDevice(Operation *op) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!targetAttr) return std::nullopt;
  return getConfigAMDAIEDevice(targetAttr);
}

/// Utility to retrieve a constant index from an OpFoldResult.
int64_t getConstantIndexOrAssert(OpFoldResult ofr) {
  std::optional<int64_t> res = getConstantIntValue(ofr);
  assert(res.has_value() && "expect constant index");
  return res.value();
}

namespace {

/// Generate a DenseMap key we can use for the element types (alternatives
/// considered: implement tombstone for std::array, or use std::map instead of
/// DenseMap).
constexpr uint32_t getElementTypeKey(uint32_t a, uint32_t b, uint32_t c) {
  return a + (b << 8) + (c << 16);
}

/// Map from (LHS bitwidth, RHS bitwidth, Accumulator bitwidth) to the AIE
/// instruction size (m, n, k) for the integer types with those bitwidths.
const auto &getIntegerMatmulInstructionSizeMap() {
  // Sanity check.
  static_assert(getElementTypeKey(1, 2, 3) == 1 + 2 * 256 + 3 * 65536);

  static DenseMap<uint32_t, std::array<uint32_t, 3>> matmulIntSizes{

      // `vector<4x16xi8>`  | `vector<16x8xi4>`  | `vector<4x8xi32>`
      {getElementTypeKey(8, 4, 32), {4, 8, 16}},

      // `vector<4x8xi8>`   | `vector<8x8xi8>`   | `vector<4x8xi32>`
      {getElementTypeKey(8, 8, 32), {4, 8, 8}},

      // `vector<4x4xi16>`  | `vector<4x8xi8>`   | `vector<4x8xi32>`
      {getElementTypeKey(16, 8, 32), {4, 8, 4}},

      // `vector<4x2xi16>`  | `vector<2x8xi16>`  | `vector<4x8xi32>`
      {getElementTypeKey(16, 16, 32), {4, 8, 2}},

      // `vector<2x8xi16>`  | `vector<8x8xi8>`   | `vector<2x8xi64>`
      // `vector<4x8xi16>`  | `vector<8x4xi8>`   | `vector<4x4xi64>`
      //   choosing the first i16 x i8 -> i64 instruction (arbitrarily)
      {getElementTypeKey(16, 8, 64), {2, 8, 8}},

      // `vector<2x4xi16>`  | `vector<4x8xi16>`  | `vector<2x8xi64>`
      // `vector<4x4xi16>`  | `vector<4x4xi16>`  | `vector<4x4xi64>`
      //   choosing the first i16 x i16 -> i64 instruction (arbitrarily)
      {getElementTypeKey(16, 16, 64), {2, 8, 4}},

      // `vector<4x2xi32>`  | `vector<2x4xi16>`  | `vector<4x4xi64>`
      {getElementTypeKey(32, 16, 64), {4, 4, 2}},
  };
  return matmulIntSizes;
}
}  // namespace

FailureOr<std::array<uint32_t, 3>> getAIEIntegerMatmulInstructionSize(
    uint32_t nBitsLhs, uint32_t nBitsRhs, uint32_t nBitsAcc) {
  const auto &mapForIntTypes = getIntegerMatmulInstructionSizeMap();
  auto it =
      mapForIntTypes.find(getElementTypeKey(nBitsLhs, nBitsRhs, nBitsAcc));
  if (it == mapForIntTypes.end()) {
    return failure();
  }
  return it->second;
}

// The number of elements in a vector instruction for a given element type.
// Reference:
// https://www.xilinx.com/htmldocs/xilinx2023_2/aiengine_ml_intrinsics/intrinsics/group__intr__gpvectorop__mul__bf16xbf16.html
FailureOr<uint32_t> getAIEMacNumElements(Type inputType, Type outputType) {
  if (inputType.isInteger(8) && outputType.isInteger(32)) return 32;
  if (inputType.isInteger(16) && outputType.isInteger(32)) return 32;
  if (inputType.isBF16() && outputType.isF32()) return 16;
  return failure();
}

FailureOr<std::array<uint32_t, 3>> getAIEMatmulInstructionSize(Type elTypeLhs,
                                                               Type elTypeRhs,
                                                               Type elTypeAcc) {
  bool allFloatingPoint = isa<FloatType>(elTypeLhs) &&
                          isa<FloatType>(elTypeRhs) &&
                          isa<FloatType>(elTypeAcc);

  bool allInteger = isa<IntegerType>(elTypeLhs) &&
                    isa<IntegerType>(elTypeRhs) && isa<IntegerType>(elTypeAcc);

  if (!allInteger && !allFloatingPoint) {
    return failure();
  }

  auto nBitsLhs = elTypeLhs.getIntOrFloatBitWidth();
  auto nBitsRhs = elTypeRhs.getIntOrFloatBitWidth();
  auto nBitsAcc = elTypeAcc.getIntOrFloatBitWidth();

  if (allFloatingPoint) {
    if (nBitsLhs == 16 && nBitsRhs == 16 && nBitsAcc == 32) {
      return std::array<uint32_t, 3>{4, 4, 8};
    }
    // There is only 1 floating point case in the table (handled above).
    return failure();
  }

  assert(allInteger);

  return getAIEIntegerMatmulInstructionSize(nBitsLhs, nBitsRhs, nBitsAcc);
}

FailureOr<unsigned> getTilingScaleFactor(Type elemType) {
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  if (bitWidth % 8 != 0) return failure();
  if (bitWidth > 64) return failure();
  return 64 / bitWidth;
}

/// Utility to match iterator type and indexing map for a linalg.generic that
/// is basically implementing a matmul with 2D input/output operands.
static bool match2DLinalgGenericMatmul(linalg::LinalgOp linalgOp) {
  // Check iterator types.
  SmallVector<utils::IteratorType> matmulIteratorTypes = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};
  SmallVector<utils::IteratorType> opIteratorTypes =
      linalgOp.getIteratorTypesArray();
  if (matmulIteratorTypes != opIteratorTypes) {
    return false;
  }
  // Check indexing maps.
  ArrayAttr indexingMaps = linalgOp.getIndexingMaps();
  if (indexingMaps.size() != 3) return false;

  AffineMap map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  AffineMap map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  AffineMap map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 2 || map1.getNumResults() != 2 ||
      map2.getNumResults() != 2 || map0.getNumInputs() != 3 ||
      map1.getNumInputs() != 3 || map2.getNumInputs() != 3) {
    return false;
  }

  AffineExpr M = map2.getResult(0);
  AffineExpr N = map2.getResult(1);
  AffineExpr K = map0.getResult(1);

  auto *context = indexingMaps.getContext();
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {M, K}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {K, N}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {M, N}, context));
  auto maps = ArrayAttr::get(context, {mapA, mapB, mapC});
  return indexingMaps == maps;
}

/// Utility to match iterator type and indexing map for a linalg.generic that
/// is basically implementing a matmul with 4D input/output operands.
static bool match4DLinalgGenericMatmul(linalg::LinalgOp linalgOp) {
  // Check iterator types.
  SmallVector<utils::IteratorType> matmulIteratorTypes = {
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::reduction, utils::IteratorType::parallel,
      utils::IteratorType::parallel,  utils::IteratorType::reduction};
  SmallVector<utils::IteratorType> opIteratorTypes =
      linalgOp.getIteratorTypesArray();
  if (matmulIteratorTypes != opIteratorTypes) {
    return false;
  }
  // Check indexing maps.
  ArrayAttr indexingMaps = linalgOp.getIndexingMaps();
  if (indexingMaps.size() != 3) return false;

  AffineMap map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  AffineMap map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  AffineMap map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 4 || map1.getNumResults() != 4 ||
      map2.getNumResults() != 4 || map0.getNumInputs() != 6 ||
      map1.getNumInputs() != 6 || map2.getNumInputs() != 6) {
    return false;
  }

  // Skip the exact indexingMaps matching, because there could be different
  // dimension permutations caused by pack_transpose.
  return true;
}

/// Utility to match iterator type and indexing map for a linalg.generic that
/// is basically implementing a matmul with 6D input/output operands.
static bool match6DLinalgGenericMatmul(linalg::LinalgOp linalgOp) {
  // Check iterator types.
  SmallVector<utils::IteratorType> matmulIteratorTypes = {
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::reduction, utils::IteratorType::parallel,
      utils::IteratorType::parallel,  utils::IteratorType::reduction,
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::reduction};
  SmallVector<utils::IteratorType> opIteratorTypes =
      linalgOp.getIteratorTypesArray();
  if (matmulIteratorTypes != opIteratorTypes) {
    return false;
  }
  // Check indexing maps.
  ArrayAttr indexingMaps = linalgOp.getIndexingMaps();
  if (indexingMaps.size() != 3) return false;

  AffineMap map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  AffineMap map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  AffineMap map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 6 || map1.getNumResults() != 6 ||
      map2.getNumResults() != 6 || map0.getNumInputs() != 9 ||
      map1.getNumInputs() != 9 || map2.getNumInputs() != 9) {
    return false;
  }

  // Skip the exact indexingMaps matching, because there could be different
  // dimension permutations caused by pack_transpose.
  return true;
}

/// Returns the BlockArgument that leads to `val`. Traverses optional ext*
/// ops.
static BlockArgument checkOptionalExtOps(Value val) {
  BlockArgument blockArg;
  if (!(blockArg = dyn_cast<BlockArgument>(val))) {
    auto defOp = val.getDefiningOp();
    if (!dyn_cast_if_present<arith::ExtFOp>(defOp) &&
        !dyn_cast_if_present<arith::ExtSIOp>(defOp) &&
        !dyn_cast_if_present<arith::ExtUIOp>(defOp)) {
      return nullptr;
    }
    blockArg = dyn_cast<BlockArgument>(defOp->getOperand(0));
  }
  return blockArg;
}

/// Utility to match block body for matmul.
static bool bodyMatcherForMatmul(Value yieldVal, Block *body) {
  Operation *addOp = yieldVal.getDefiningOp();
  if (!isa_and_present<arith::AddIOp, arith::AddFOp>(addOp)) {
    return false;
  }
  Operation *mulOp = addOp->getOperand(1).getDefiningOp();
  if (!isa_and_present<arith::MulIOp, arith::MulFOp>(mulOp)) {
    return false;
  }

  BlockArgument lhsBlockArg = checkOptionalExtOps(mulOp->getOperand(0));
  BlockArgument rhsBlockArg = checkOptionalExtOps(mulOp->getOperand(1));
  BlockArgument outBlockArg = checkOptionalExtOps(addOp->getOperand(0));
  if (!lhsBlockArg || !rhsBlockArg || !outBlockArg ||
      lhsBlockArg.getOwner() != body || rhsBlockArg.getOwner() != body ||
      outBlockArg.getOwner() != body || lhsBlockArg.getArgNumber() != 0 ||
      rhsBlockArg.getArgNumber() != 1 || outBlockArg.getArgNumber() != 2) {
    return false;
  }
  return true;
}

/// Utility to indentify whether a linalg op is a matmul op.
bool isMatmul(linalg::LinalgOp linalgOp) {
  // Step 0. Test if the op itself is a linalg.matmul op.
  if (isa<linalg::MatmulOp>(linalgOp)) return true;

  // Step 1. Test the body of the generic to indeed be what we expect for a
  //         matmul.
  Block *body = linalgOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  if (!bodyMatcherForMatmul(yieldVal, body)) {
    return false;
  }

  return match2DLinalgGenericMatmul(linalgOp) ||
         match4DLinalgGenericMatmul(linalgOp) ||
         match6DLinalgGenericMatmul(linalgOp);
}

/// Utility to identify if the input operand has matmul-like op in its
/// def-chain.
bool isMatmulInDefChain(Value operand) {
  Operation *defOp = operand.getDefiningOp();
  if (!defOp) {
    return false;
  }

  if (isa<arith::ConstantOp>(defOp)) {
    return false;
  }

  if (auto defLinalgOp = dyn_cast_if_present<linalg::LinalgOp>(defOp)) {
    if (isMatmul(defLinalgOp)) {
      return true;
    }
  }

  // If something is being produced from a for/forall loop, we just assume it is
  // some fused computation and do not really need to look at its body to match
  // matmul.
  if (isa<scf::ForOp>(defOp) || isa<scf::ForallOp>(defOp)) {
    return true;
  }

  for (Value operand : defOp->getOperands()) {
    if (isMatmulInDefChain(operand)) {
      return true;
    }
  }
  return false;
}

/// Utility to identify if `linalgOp` is an elementwise operation with a
/// matmul-like op upstream in its computation tree.
bool isMatmulProducerOfElementwise(linalg::LinalgOp linalgOp) {
  if (!isElementwise(linalgOp) || isa<linalg::FillOp>(linalgOp)) {
    return false;
  }
  // Check if any of the defining op is a matmul-like op.
  for (Value operand : linalgOp->getOperands()) {
    if (isMatmulInDefChain(operand)) {
      return true;
    }
  }
  return false;
}

/// Find the largest factor of 'num' which is not larger than 'max'.
int detail::findLargestFactor(int num, int max) {
  assert(max > 0 && "No factors less than or equal to 0 exist");

  // Do O(1) instead of O(sqrt(num)) computation for this common case.
  if (num <= max) {
    return num;
  }

  int largestLowFactor = 1;
  for (int lowFactor = 2; lowFactor <= max; ++lowFactor) {
    const int highFactor = num / lowFactor;

    // This early exit is what makes this O(sqrt(num)) instead of O(num).
    if (highFactor < lowFactor) return largestLowFactor;

    const bool areActuallyFactors = num % lowFactor == 0;
    if (areActuallyFactors) {
      // We're certain that here lowFactor <= highFactor, and highFactor is
      // descending in this loop. So we can return immediately if highFactor is
      // good.
      if (highFactor <= max) return highFactor;
      largestLowFactor = lowFactor;
    }
  }
  return largestLowFactor;
}

/// Find the largest factor of 'num' which is not larger than 'max' and is a
/// multiple of `multiple` if possible.
int detail::findLargestFactor(int num, int max, int multiple) {
  int factor = 0;
  for (int i = multiple; i <= max && i <= num; i += multiple) {
    if (num % i == 0 && i % multiple == 0) {
      factor = i;
    }
  }
  // if we could not find the desired factor then we give up and call the code
  // that doesnt require the multiple constrain.
  return factor ? factor : detail::findLargestFactor(num, max);
}

}  // namespace mlir::iree_compiler::AMDAIE
