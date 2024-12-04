// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gtest/gtest.h"
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace {

using namespace mlir;
using namespace mlir::iree_compiler::AMDAIE;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class AccessPatternCombinationTest : public ::testing::Test {
 protected:
  AccessPatternCombinationTest()
      : rewriter(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<arith::ArithDialect>();
  }

  SmallVector<OpFoldResult> toOpFoldResult(const SmallVector<int64_t> &values) {
    return llvm::map_to_vector(values, [&](int64_t v) -> OpFoldResult {
      return rewriter.getI64IntegerAttr(v);
    });
  }

  bool checkAreAccessPatternsCombinable(const SmallVector<int64_t> &offsetsA,
                                        const SmallVector<int64_t> &sizesA,
                                        const SmallVector<int64_t> &stridesA,
                                        const SmallVector<int64_t> &offsetsB,
                                        const SmallVector<int64_t> &sizesB,
                                        const SmallVector<int64_t> &stridesB,
                                        size_t maxNbDims) {
    SmallVector<OpFoldResult> offsetsValuesA = toOpFoldResult(offsetsA);
    SmallVector<OpFoldResult> sizesValuesA = toOpFoldResult(sizesA);
    SmallVector<OpFoldResult> stridesValuesA = toOpFoldResult(stridesA);
    SmallVector<OpFoldResult> offsetsValuesB = toOpFoldResult(offsetsB);
    SmallVector<OpFoldResult> sizesValuesB = toOpFoldResult(sizesB);
    SmallVector<OpFoldResult> stridesValuesB = toOpFoldResult(stridesB);
    return areAccessPatternsCombinable(offsetsValuesA, sizesValuesA,
                                       stridesValuesA, offsetsValuesB,
                                       sizesValuesB, stridesValuesB, maxNbDims);
  }

  void checkCombineAccessPatterns(const SmallVector<int64_t> offsetsA,
                                  const SmallVector<int64_t> sizesA,
                                  const SmallVector<int64_t> stridesA,
                                  const SmallVector<int64_t> offsetsB,
                                  const SmallVector<int64_t> sizesB,
                                  const SmallVector<int64_t> stridesB,
                                  const SmallVector<int64_t> expectedOffsets,
                                  const SmallVector<int64_t> expectedSizes,
                                  const SmallVector<int64_t> expectedStrides,
                                  size_t maxNbDims, bool shouldSucceed = true) {
    SmallVector<OpFoldResult> offsetsValuesA = toOpFoldResult(offsetsA);
    SmallVector<OpFoldResult> sizesValuesA = toOpFoldResult(sizesA);
    SmallVector<OpFoldResult> stridesValuesA = toOpFoldResult(stridesA);
    SmallVector<OpFoldResult> offsetsValuesB = toOpFoldResult(offsetsB);
    SmallVector<OpFoldResult> sizesValuesB = toOpFoldResult(sizesB);
    SmallVector<OpFoldResult> stridesValuesB = toOpFoldResult(stridesB);
    SmallVector<OpFoldResult> expectedOffsetsValues =
        toOpFoldResult(expectedOffsets);
    SmallVector<OpFoldResult> expectedSizesValues =
        toOpFoldResult(expectedSizes);
    SmallVector<OpFoldResult> expectedStridesValues =
        toOpFoldResult(expectedStrides);
    SmallVector<OpFoldResult> newOffsets;
    SmallVector<OpFoldResult> newSizes;
    SmallVector<OpFoldResult> newStrides;
    if (shouldSucceed) {
      EXPECT_TRUE(succeeded(combineAccessPatterns(
          rewriter, offsetsValuesA, sizesValuesA, stridesValuesA,
          offsetsValuesB, sizesValuesB, stridesValuesB, newOffsets, newSizes,
          newStrides, maxNbDims)));
      EXPECT_EQ(newOffsets, expectedOffsetsValues);
      EXPECT_EQ(newSizes, expectedSizesValues);
      EXPECT_EQ(newStrides, expectedStridesValues);
    } else {
      EXPECT_TRUE(failed(combineAccessPatterns(
          rewriter, offsetsValuesA, sizesValuesA, stridesValuesA,
          offsetsValuesB, sizesValuesB, stridesValuesB, newOffsets, newSizes,
          newStrides, maxNbDims)));
    }
  }

  MLIRContext context;
  IRRewriter rewriter;
  Location loc;
};

TEST_F(AccessPatternCombinationTest, CombinableAccessPatterns) {
  EXPECT_TRUE(checkAreAccessPatternsCombinable({}, {}, {}, {}, {}, {}, 1));
  // size(A) == size(B)
  EXPECT_TRUE(
      checkAreAccessPatternsCombinable({0}, {16}, {1}, {32}, {16}, {1}, 2));
  EXPECT_TRUE(checkAreAccessPatternsCombinable({0, 0}, {16, 32}, {64, 1},
                                               {0, 32}, {16, 32}, {64, 1}, 4));
  EXPECT_TRUE(checkAreAccessPatternsCombinable({1, 0}, {16, 32}, {64, 1},
                                               {1, 32}, {16, 32}, {64, 1}, 4));
  EXPECT_TRUE(checkAreAccessPatternsCombinable({0, 0, 0}, {16, 16, 32},
                                               {32, 64, 1}, {0, 0, 32},
                                               {16, 16, 32}, {32, 64, 1}, 4));
  EXPECT_TRUE(checkAreAccessPatternsCombinable({0, 2, 0}, {16, 16, 32},
                                               {32, 64, 1}, {0, 2, 32},
                                               {16, 16, 32}, {32, 64, 1}, 4));
  EXPECT_TRUE(checkAreAccessPatternsCombinable({32, 0}, {64, 64}, {128, 1},
                                               {96, 0}, {32, 64}, {128, 1}, 4));
  // size(A) > size(B)
  EXPECT_TRUE(checkAreAccessPatternsCombinable(
      {0, 0, 0}, {2, 16, 32}, {32, 64, 1}, {0, 64}, {16, 32}, {64, 1}, 4));
  EXPECT_TRUE(checkAreAccessPatternsCombinable(
      {0, 0, 0}, {2, 16, 32}, {32, 64, 1}, {1, 0}, {16, 32}, {64, 1}, 4));
  EXPECT_TRUE(checkAreAccessPatternsCombinable(
      {0, 0, 32}, {2, 16, 32}, {32, 64, 1}, {0, 96}, {16, 32}, {64, 1}, 4));
  EXPECT_TRUE(checkAreAccessPatternsCombinable(
      {0, 2, 0}, {2, 16, 32}, {32, 16, 1}, {6, 0}, {16, 32}, {16, 1}, 4));
  // size(B) > size(A)
  EXPECT_TRUE(checkAreAccessPatternsCombinable(
      {0, 0}, {16, 32}, {64, 1}, {0, 0, 32}, {2, 16, 32}, {32, 64, 1}, 4));
  EXPECT_TRUE(checkAreAccessPatternsCombinable(
      {0, 0}, {16, 32}, {16, 1}, {0, 2, 0}, {2, 16, 32}, {32, 16, 1}, 4));
  EXPECT_TRUE(checkAreAccessPatternsCombinable(
      {0, 32}, {16, 32}, {64, 1}, {0, 0, 64}, {2, 16, 32}, {32, 64, 1}, 4));
  EXPECT_TRUE(checkAreAccessPatternsCombinable(
      {2, 0}, {16, 32}, {16, 1}, {0, 4, 0}, {2, 16, 32}, {32, 16, 1}, 4));
}

TEST_F(AccessPatternCombinationTest, NonCombinableAccessPatterns) {
  // |size(A) - size(B)| > 1
  EXPECT_FALSE(checkAreAccessPatternsCombinable({}, {}, {}, {0, 0}, {16, 32},
                                                {64, 1}, 3));
  EXPECT_FALSE(checkAreAccessPatternsCombinable({0}, {32}, {1}, {0, 0, 32},
                                                {2, 16, 32}, {128, 64, 1}, 3));
  EXPECT_FALSE(checkAreAccessPatternsCombinable({0, 0}, {16, 32}, {64, 1}, {},
                                                {}, {}, 3));
  EXPECT_FALSE(checkAreAccessPatternsCombinable(
      {0, 0, 32}, {2, 16, 32}, {128, 64, 1}, {0}, {32}, {1}, 3));
  // Same access patterns
  EXPECT_FALSE(
      checkAreAccessPatternsCombinable({0}, {32}, {1}, {0}, {32}, {1}, 2));
  EXPECT_FALSE(checkAreAccessPatternsCombinable({0, 0}, {16, 32}, {64, 1},
                                                {0, 0}, {16, 32}, {64, 1}, 4));
  // Too few dimensions
  EXPECT_FALSE(
      checkAreAccessPatternsCombinable({0}, {16}, {1}, {32}, {16}, {1}, 1));
  EXPECT_FALSE(checkAreAccessPatternsCombinable({0, 0}, {16, 32}, {64, 1},
                                                {0, 32}, {16, 32}, {64, 1}, 2));
  EXPECT_FALSE(checkAreAccessPatternsCombinable({0, 0, 0}, {16, 16, 32},
                                                {32, 64, 1}, {0, 0, 32},
                                                {16, 16, 32}, {32, 64, 1}, 3));
  // size(A) > size(B) Incompatible offset
  EXPECT_FALSE(checkAreAccessPatternsCombinable(
      {0, 0, 0}, {2, 16, 32}, {32, 64, 1}, {0, 32}, {16, 32}, {64, 1}, 4));
  EXPECT_FALSE(checkAreAccessPatternsCombinable(
      {0, 0, 0}, {2, 16, 32}, {32, 64, 1}, {0, 128}, {16, 32}, {64, 1}, 4));
  EXPECT_FALSE(checkAreAccessPatternsCombinable(
      {0, 0, 0}, {2, 16, 32}, {32, 64, 1}, {64, 0}, {16, 32}, {64, 1}, 4));
  // size(B) > size(A) Incompatible offset
  EXPECT_FALSE(checkAreAccessPatternsCombinable(
      {0, 0}, {16, 32}, {64, 1}, {0, 0, 16}, {2, 16, 32}, {32, 64, 1}, 4));
  EXPECT_FALSE(checkAreAccessPatternsCombinable(
      {0, 0}, {16, 32}, {64, 1}, {0, 0, 96}, {2, 16, 32}, {32, 64, 1}, 4));
  EXPECT_FALSE(checkAreAccessPatternsCombinable(
      {0, 0}, {16, 32}, {64, 1}, {0, 1, 0}, {2, 16, 32}, {32, 64, 1}, 4));

  // size(A) == size(B) Incompatible offset
  EXPECT_FALSE(checkAreAccessPatternsCombinable(
      {32, 0}, {64, 64}, {128, 1}, {32, 0}, {32, 64}, {128, 1}, 4));
  EXPECT_FALSE(checkAreAccessPatternsCombinable(
      {32, 0}, {32, 64}, {128, 1}, {96, 0}, {64, 64}, {128, 1}, 4));
}

TEST_F(AccessPatternCombinationTest, CombineAccessPatterns) {
  checkCombineAccessPatterns({}, {}, {}, {}, {}, {}, {}, {}, {}, 1);
  // size(A) == size(B)
  checkCombineAccessPatterns({0}, {16}, {1}, {32}, {16}, {1}, {0, 0}, {2, 16},
                             {32, 1}, 2);
  checkCombineAccessPatterns({0, 0}, {8, 16}, {8, 1}, {0, 32}, {8, 16}, {8, 1},
                             {0, 0, 0}, {2, 8, 16}, {32, 8, 1}, 3);
  checkCombineAccessPatterns({0, 32}, {8, 16}, {8, 1}, {0, 64}, {8, 16}, {8, 1},
                             {0, 0, 32}, {2, 8, 16}, {32, 8, 1}, 3);
  checkCombineAccessPatterns({1, 32}, {8, 16}, {8, 1}, {1, 64}, {8, 16}, {8, 1},
                             {0, 1, 32}, {2, 8, 16}, {32, 8, 1}, 3);
  checkCombineAccessPatterns({0, 0}, {8, 16}, {8, 1}, {32, 0}, {8, 16}, {8, 1},
                             {0, 0, 0}, {2, 8, 16}, {256, 8, 1}, 3);
  checkCombineAccessPatterns({8, 0}, {8, 16}, {8, 1}, {40, 0}, {8, 16}, {8, 1},
                             {0, 8, 0}, {2, 8, 16}, {256, 8, 1}, 3);
  checkCombineAccessPatterns({0, 0, 0}, {16, 8, 16}, {16, 8, 1}, {0, 0, 32},
                             {16, 8, 16}, {16, 8, 1}, {0, 0, 0, 0},
                             {2, 16, 8, 16}, {32, 16, 8, 1}, 4);
  checkCombineAccessPatterns({0, 0, 32}, {16, 8, 16}, {16, 8, 1}, {0, 0, 64},
                             {16, 8, 16}, {16, 8, 1}, {0, 0, 0, 32},
                             {2, 16, 8, 16}, {32, 16, 8, 1}, 4);
  checkCombineAccessPatterns({0, 0, 0}, {16, 8, 16}, {16, 8, 1}, {32, 0, 0},
                             {16, 8, 16}, {16, 8, 1}, {0, 0, 0, 0},
                             {2, 16, 8, 16}, {512, 16, 8, 1}, 4);
  checkCombineAccessPatterns({8, 0, 0}, {16, 8, 16}, {16, 8, 1}, {40, 0, 0},
                             {16, 8, 16}, {16, 8, 1}, {0, 8, 0, 0},
                             {2, 16, 8, 16}, {512, 16, 8, 1}, 4);
  checkCombineAccessPatterns({32, 0}, {64, 64}, {128, 1}, {96, 0}, {32, 64},
                             {128, 1}, {32, 0}, {96, 64}, {128, 1}, 4);
  // size(A) > size(B)
  checkCombineAccessPatterns({0, 0}, {2, 32}, {64, 1}, {128}, {32}, {1}, {0, 0},
                             {3, 32}, {64, 1}, 3);
  checkCombineAccessPatterns({0, 32}, {3, 32}, {64, 1}, {224}, {32}, {1},
                             {0, 32}, {4, 32}, {64, 1}, 3);
  checkCombineAccessPatterns({0, 0, 0}, {2, 16, 32}, {32, 64, 1}, {0, 64},
                             {16, 32}, {64, 1}, {0, 0, 0}, {3, 16, 32},
                             {32, 64, 1}, 4);
  checkCombineAccessPatterns({0, 0, 0}, {2, 16, 32}, {32, 64, 1}, {1, 0},
                             {16, 32}, {64, 1}, {0, 0, 0}, {3, 16, 32},
                             {32, 64, 1}, 4);
  checkCombineAccessPatterns({0, 1, 0}, {2, 16, 32}, {32, 64, 1}, {2, 0},
                             {16, 32}, {64, 1}, {0, 1, 0}, {3, 16, 32},
                             {32, 64, 1}, 4);
  checkCombineAccessPatterns({0, 1, 32}, {2, 16, 32}, {32, 64, 1}, {2, 32},
                             {16, 32}, {64, 1}, {0, 1, 32}, {3, 16, 32},
                             {32, 64, 1}, 4);
  // size(B) > size(A)
  checkCombineAccessPatterns({0}, {32}, {1}, {0, 64}, {2, 32}, {64, 1}, {0, 0},
                             {3, 32}, {64, 1}, 3);
  checkCombineAccessPatterns({32}, {32}, {1}, {0, 96}, {2, 32}, {64, 1},
                             {0, 32}, {3, 32}, {64, 1}, 3);
  checkCombineAccessPatterns({0, 0}, {16, 32}, {16, 1}, {0, 0, 64}, {2, 16, 32},
                             {64, 16, 1}, {0, 0, 0}, {3, 16, 32}, {64, 16, 1},
                             4);
  checkCombineAccessPatterns({0, 32}, {16, 32}, {16, 1}, {0, 0, 96},
                             {2, 16, 32}, {64, 16, 1}, {0, 0, 32}, {3, 16, 32},
                             {64, 16, 1}, 4);
  checkCombineAccessPatterns({2, 0}, {16, 32}, {16, 1}, {0, 6, 0}, {2, 16, 32},
                             {64, 16, 1}, {0, 2, 0}, {3, 16, 32}, {64, 16, 1},
                             4);
  checkCombineAccessPatterns({2, 32}, {16, 32}, {16, 1}, {0, 6, 32},
                             {2, 16, 32}, {64, 16, 1}, {0, 2, 32}, {3, 16, 32},
                             {64, 16, 1}, 4);
}

TEST_F(AccessPatternCombinationTest, FailCombineAccessPatterns) {
  // |size(A) - size(B)| > 1
  checkCombineAccessPatterns({}, {}, {}, {0, 0}, {16, 32}, {16, 1}, {}, {}, {},
                             3, false);
  checkCombineAccessPatterns({0, 0}, {16, 32}, {16, 1}, {}, {}, {}, {}, {}, {},
                             3, false);
  // Same access pattern
  checkCombineAccessPatterns({0, 0}, {16, 32}, {16, 1}, {0, 0}, {16, 32},
                             {16, 1}, {}, {}, {}, 3, false);
  // Too few dimensions
  checkCombineAccessPatterns({0, 0}, {16, 32}, {16, 1}, {0, 32}, {16, 32},
                             {16, 1}, {}, {}, {}, 2, false);
  // size(A) > size(B) Incompatible offset
  checkCombineAccessPatterns({0, 0}, {2, 32}, {64, 1}, {96}, {32}, {1}, {0, 0},
                             {3, 32}, {64, 1}, 3, false);
  checkCombineAccessPatterns({0, 0}, {2, 32}, {64, 1}, {256}, {32}, {1}, {0, 0},
                             {3, 32}, {64, 1}, 3, false);
  // size(B) > size(A) Incompatible offset
  checkCombineAccessPatterns({0}, {32}, {1}, {0, 32}, {2, 32}, {64, 1}, {0, 0},
                             {3, 32}, {64, 1}, 3, false);
  checkCombineAccessPatterns({0}, {32}, {1}, {0, 96}, {2, 32}, {64, 1}, {0, 0},
                             {3, 32}, {64, 1}, 3, false);

  // size(A) == size(B) Incompatible offset
  checkCombineAccessPatterns({32, 0}, {32, 64}, {128, 1}, {96, 0}, {64, 64},
                             {128, 1}, {32, 0}, {96, 64}, {128, 1}, 4, false);
}

class FoldTest : public ::testing::Test {
 protected:
  FoldTest() : rewriter(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<arith::ArithDialect>();
  }

  SmallVector<OpFoldResult> toOpFoldResult(ArrayRef<int64_t> values) {
    return llvm::map_to_vector(values, [&](int64_t v) -> OpFoldResult {
      return getAsIndexOpFoldResult(&context, v);
    });
  }

  void checkFoldLinearDims(const SmallVector<int64_t> offsets,
                           const SmallVector<int64_t> sizes,
                           const SmallVector<int64_t> strides,
                           ArrayRef<int64_t> maxSizes,
                           const SmallVector<int64_t> expectedOffsets,
                           const SmallVector<int64_t> expectedSizes,
                           const SmallVector<int64_t> expectedStrides,
                           bool shouldSucceed = true) {
    SmallVector<OpFoldResult> offsetsValues = toOpFoldResult(offsets);
    SmallVector<OpFoldResult> sizesValues = toOpFoldResult(sizes);
    SmallVector<OpFoldResult> stridesValues = toOpFoldResult(strides);
    SmallVector<OpFoldResult> expectedOffsetsValues =
        toOpFoldResult(expectedOffsets);
    SmallVector<OpFoldResult> expectedSizesValues =
        toOpFoldResult(expectedSizes);
    SmallVector<OpFoldResult> expectedStridesValues =
        toOpFoldResult(expectedStrides);
    SmallVector<OpFoldResult> newOffsets;
    SmallVector<OpFoldResult> newSizes;
    SmallVector<OpFoldResult> newStrides;
    if (shouldSucceed) {
      EXPECT_TRUE(succeeded(foldLinearDims(&context, offsetsValues, sizesValues,
                                           stridesValues, newOffsets, newSizes,
                                           newStrides, maxSizes)));
    } else {
      EXPECT_TRUE(failed(foldLinearDims(&context, offsetsValues, sizesValues,
                                        stridesValues, newOffsets, newSizes,
                                        newStrides, maxSizes)));
    }
    EXPECT_EQ(newOffsets, expectedOffsetsValues);
    EXPECT_EQ(newSizes, expectedSizesValues);
    EXPECT_EQ(newStrides, expectedStridesValues);
  }

  void checkFoldUnitDims(const SmallVector<int64_t> offsets,
                         const SmallVector<int64_t> sizes,
                         const SmallVector<int64_t> strides,
                         const SmallVector<int64_t> expectedOffsets,
                         const SmallVector<int64_t> expectedSizes,
                         const SmallVector<int64_t> expectedStrides,
                         bool shouldSucceed = true) {
    SmallVector<OpFoldResult> offsetsValues = toOpFoldResult(offsets);
    SmallVector<OpFoldResult> sizesValues = toOpFoldResult(sizes);
    SmallVector<OpFoldResult> stridesValues = toOpFoldResult(strides);
    SmallVector<OpFoldResult> expectedOffsetsValues =
        toOpFoldResult(expectedOffsets);
    SmallVector<OpFoldResult> expectedSizesValues =
        toOpFoldResult(expectedSizes);
    SmallVector<OpFoldResult> expectedStridesValues =
        toOpFoldResult(expectedStrides);
    SmallVector<OpFoldResult> newOffsets;
    SmallVector<OpFoldResult> newSizes;
    SmallVector<OpFoldResult> newStrides;
    if (shouldSucceed) {
      EXPECT_TRUE(succeeded(foldUnitDims(&context, offsetsValues, sizesValues,
                                         stridesValues, newOffsets, newSizes,
                                         newStrides)));
      EXPECT_EQ(newOffsets, expectedOffsetsValues);
      EXPECT_EQ(newSizes, expectedSizesValues);
      EXPECT_EQ(newStrides, expectedStridesValues);
    } else {
      EXPECT_TRUE(failed(foldUnitDims(&context, offsetsValues, sizesValues,
                                      stridesValues, newOffsets, newSizes,
                                      newStrides)));
    }
  }

  MLIRContext context;
  IRRewriter rewriter;
  Location loc;
};

TEST_F(FoldTest, NoLinearDimsFold) {
  checkFoldLinearDims({}, {}, {}, {}, {}, {}, {}, false);
  checkFoldLinearDims({0}, {8}, {1}, {}, {0}, {8}, {1}, false);
  checkFoldLinearDims({0, 0}, {16, 8}, {16, 1}, {}, {0, 0}, {16, 8}, {16, 1},
                      false);
  checkFoldLinearDims({8, 0}, {16, 8}, {8, 1}, {}, {8, 0}, {16, 8}, {8, 1},
                      false);
}

TEST_F(FoldTest, FoldLinearDims) {
  checkFoldLinearDims({0, 0}, {16, 8}, {8, 1}, {}, {0}, {128}, {1}, true);
  checkFoldLinearDims({0, 8}, {16, 8}, {8, 1}, {}, {8}, {128}, {1}, true);
  checkFoldLinearDims({0, 0, 0}, {8, 16, 8}, {128, 8, 1}, {}, {0}, {1024}, {1},
                      true);
  checkFoldLinearDims({0, 0, 0, 0}, {4, 8, 16, 8}, {1024, 128, 8, 1}, {}, {0},
                      {4096}, {1}, true);
  checkFoldLinearDims({0, 0, 8, 0}, {4, 8, 16, 8}, {1024, 128, 8, 1}, {},
                      {8, 0}, {512, 8}, {8, 1}, true);
}

TEST_F(FoldTest, FoldLinearDimsWithMax) {
  checkFoldLinearDims({0, 0}, {16, 8}, {8, 1}, {127}, {0, 0}, {16, 8}, {8, 1},
                      false);
  checkFoldLinearDims({0, 0}, {16, 8}, {8, 1}, {127, 127}, {0, 0}, {16, 8},
                      {8, 1}, false);
  checkFoldLinearDims({0, 0}, {16, 8}, {8, 1}, {128}, {0}, {128}, {1}, true);
  checkFoldLinearDims({0, 0, 0}, {8, 16, 8}, {128, 8, 1}, {1023, 1023, 1023},
                      {0, 0}, {8, 128}, {128, 1}, true);
  checkFoldLinearDims({0, 0, 0, 0}, {4, 8, 16, 8}, {1024, 128, 8, 1},
                      {1024, 1024, 1024, 1024}, {0, 0}, {4, 1024}, {1024, 1},
                      true);
  checkFoldLinearDims({0, 0, 8, 0}, {4, 8, 16, 8}, {1024, 128, 8, 1},
                      {511, 511, 511, 511}, {0, 8, 0}, {4, 128, 8},
                      {1024, 8, 1}, true);
}

TEST_F(FoldTest, NoUnitDimsFold) {
  checkFoldUnitDims({}, {}, {}, {}, {}, {}, false);
  checkFoldUnitDims({0}, {8}, {1}, {}, {}, {}, false);
  checkFoldUnitDims({0, 0}, {16, 8}, {16, 1}, {}, {}, {}, false);
  checkFoldUnitDims({2}, {1}, {1}, {}, {}, {}, false);
}

TEST_F(FoldTest, UnitDimsFullFold) {
  checkFoldUnitDims({0}, {1}, {32}, {}, {}, {}, true);
  checkFoldUnitDims({0, 0, 0}, {32, 1, 8}, {32, 1024, 1}, {0, 0}, {32, 8},
                    {32, 1}, true);
  checkFoldUnitDims({0, 0, 0, 0}, {1, 32, 1, 8}, {1024, 32, 1024, 1}, {0, 0},
                    {32, 8}, {32, 1}, true);
}

TEST_F(FoldTest, UnitDimsMerge) {
  checkFoldUnitDims({1, 1}, {1, 1}, {32, 32}, {2}, {1}, {32}, true);
  checkFoldUnitDims({1, 2}, {1, 1}, {32, 32}, {3}, {1}, {32}, true);
  checkFoldUnitDims({2, 1}, {1, 1}, {32, 32}, {3}, {1}, {32}, true);
  checkFoldUnitDims({1, 0, 1, 0}, {1, 32, 1, 8}, {1024, 32, 1024, 1}, {2, 0, 0},
                    {1, 32, 8}, {1024, 32, 1}, true);
  checkFoldUnitDims({1, 0, 2, 0}, {1, 32, 1, 8}, {1024, 32, 1024, 1}, {3, 0, 0},
                    {1, 32, 8}, {1024, 32, 1}, true);
  checkFoldUnitDims({2, 0, 1, 0}, {1, 32, 1, 8}, {1024, 32, 1024, 1}, {3, 0, 0},
                    {1, 32, 8}, {1024, 32, 1}, true);
}

TEST_F(FoldTest, UnitDimsFoldAndMerge) {
  checkFoldUnitDims({1, 0, 1}, {1, 1, 1}, {32, 1024, 32}, {2}, {1}, {32}, true);
  checkFoldUnitDims({1, 0, 1}, {1, 1, 1}, {32, 32, 32}, {2}, {1}, {32}, true);
  checkFoldUnitDims({1, 0, 2, 0}, {1, 1, 1, 1}, {32, 32, 32, 32}, {3}, {1},
                    {32}, true);
  checkFoldUnitDims({1, 0, 1, 0}, {1, 1, 1, 8}, {1024, 32, 1024, 1}, {2, 0},
                    {1, 8}, {1024, 1}, true);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
