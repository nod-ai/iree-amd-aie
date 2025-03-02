// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gtest/gtest.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace {

using namespace mlir;
using namespace mlir::iree_compiler::AMDAIE;
using namespace mlir::iree_compiler::AMDAIE::detail;

SmallVector<int64_t> fromOpFoldResults(SmallVector<OpFoldResult> ofrs) {
  std::optional<SmallVector<int64_t>> vals = getConstantIntValues(ofrs);
  assert(vals.has_value() && "expected constant values");
  return vals.value();
}

//===----------------------------------------------------------------------===//
// Detail Tests
//===----------------------------------------------------------------------===//

TEST(FindFactorResultingInSmallerSize, TestEarlyExit) {
  EXPECT_EQ(findFactorResultingInSmallerSize(0, 0), 1);
  EXPECT_EQ(findFactorResultingInSmallerSize(0, 1), 1);
  EXPECT_EQ(findFactorResultingInSmallerSize(1, 0), 1);
  EXPECT_EQ(findFactorResultingInSmallerSize(1, 1), 1);
  // size <= maxSize
  EXPECT_EQ(findFactorResultingInSmallerSize(1, 2), 1);
  EXPECT_EQ(findFactorResultingInSmallerSize(2, 2), 1);
  EXPECT_EQ(findFactorResultingInSmallerSize(1023, 1024), 1);
}

TEST(FindFactorResultingInSmallerSize, TestMain) {
  EXPECT_EQ(findFactorResultingInSmallerSize(3, 2), 3);
  EXPECT_EQ(findFactorResultingInSmallerSize(4, 2), 2);
  EXPECT_EQ(findFactorResultingInSmallerSize(5, 2), 5);
  EXPECT_EQ(findFactorResultingInSmallerSize(6, 2), 3);
  EXPECT_EQ(findFactorResultingInSmallerSize(9, 2), 9);
  EXPECT_EQ(findFactorResultingInSmallerSize(12, 2), 6);
  EXPECT_EQ(findFactorResultingInSmallerSize(4, 3), 2);
  EXPECT_EQ(findFactorResultingInSmallerSize(5, 3), 5);
  EXPECT_EQ(findFactorResultingInSmallerSize(6, 3), 2);
  EXPECT_EQ(findFactorResultingInSmallerSize(7, 3), 7);
  EXPECT_EQ(findFactorResultingInSmallerSize(8, 3), 4);
  EXPECT_EQ(findFactorResultingInSmallerSize(9, 3), 3);
}

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class AccessPatternCombinationTest : public ::testing::Test {
 protected:
  AccessPatternCombinationTest()
      : rewriter(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<arith::ArithDialect>();
  }

  SmallVector<OpFoldResult> toOpFoldResults(SmallVector<int64_t> values) {
    return llvm::map_to_vector(values, [&](int64_t v) -> OpFoldResult {
      return getAsIndexOpFoldResult(&context, v);
    });
  }

  bool checkAccessPaternsCombinable(SmallVector<int64_t> offsetsA,
                                    SmallVector<int64_t> sizesA,
                                    SmallVector<int64_t> stridesA,
                                    SmallVector<int64_t> offsetsB,
                                    SmallVector<int64_t> sizesB,
                                    SmallVector<int64_t> stridesB,
                                    function_ref<bool(size_t)> exceedsNbDims) {
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    return succeeded(combineAccessPatterns(
        &context, toOpFoldResults(offsetsA), toOpFoldResults(sizesA),
        toOpFoldResults(stridesA), toOpFoldResults(offsetsB),
        toOpFoldResults(sizesB), toOpFoldResults(stridesB), newOffsets,
        newSizes, newStrides, exceedsNbDims));
  }

  bool checkAccessPaternsCombinable(SmallVector<int64_t> offsetsA,
                                    SmallVector<int64_t> sizesA,
                                    SmallVector<int64_t> stridesA,
                                    SmallVector<int64_t> offsetsB,
                                    SmallVector<int64_t> sizesB,
                                    SmallVector<int64_t> stridesB,
                                    size_t maxNbDims) {
    auto rankTooLarge = [&](size_t r) { return r > maxNbDims; };
    return checkAccessPaternsCombinable(offsetsA, sizesA, stridesA, offsetsB,
                                        sizesB, stridesB, rankTooLarge);
  }

  bool checkCombine(SmallVector<int64_t> offsetsA, SmallVector<int64_t> sizesA,
                    SmallVector<int64_t> stridesA,
                    SmallVector<int64_t> offsetsB, SmallVector<int64_t> sizesB,
                    SmallVector<int64_t> stridesB,
                    SmallVector<int64_t> expectedOffsets,
                    SmallVector<int64_t> expectedSizes,
                    SmallVector<int64_t> expectedStrides, size_t maxNbDims,
                    bool checkValues = true) {
    SmallVector<OpFoldResult> newOffsets;
    SmallVector<OpFoldResult> newSizes;
    SmallVector<OpFoldResult> newStrides;

    auto success = combineAccessPatterns(
        &context, toOpFoldResults(offsetsA), toOpFoldResults(sizesA),
        toOpFoldResults(stridesA), toOpFoldResults(offsetsB),
        toOpFoldResults(sizesB), toOpFoldResults(stridesB), newOffsets,
        newSizes, newStrides, [&](size_t dim) { return dim > maxNbDims; });
    if (checkValues) {
      EXPECT_EQ(fromOpFoldResults(newOffsets), expectedOffsets);
      EXPECT_EQ(fromOpFoldResults(newSizes), expectedSizes);
      EXPECT_EQ(fromOpFoldResults(newStrides), expectedStrides);
    }
    return succeeded(success);
  }

  int64_t getGlobalOffsetDifference(SmallVector<int64_t> offsetsX,
                                    SmallVector<int64_t> stridesX,
                                    SmallVector<int64_t> offsetsY,
                                    SmallVector<int64_t> stridesY) {
    std::optional<int64_t> goDiff =
        mlir::iree_compiler::AMDAIE::detail::getGlobalOffsetDifference(
            toOpFoldResults(offsetsX), toOpFoldResults(stridesX),
            toOpFoldResults(offsetsY), toOpFoldResults(stridesY));

    EXPECT_TRUE(goDiff.has_value());
    if (goDiff.has_value()) return goDiff.value();

    return -1;
  }

  MLIRContext context;
  IRRewriter rewriter;
  Location loc;
};

// This test checks correctness in the case where all inputs (offsets and
// strides) are constant. The cases where they are mlir Values are tested in the
// lit testing.
TEST_F(AccessPatternCombinationTest, GlobalOffsetTest) {
  EXPECT_EQ(getGlobalOffsetDifference({}, {}, {}, {}), 0);
  EXPECT_EQ(getGlobalOffsetDifference({1}, {1}, {2}, {3}), 1 * 1 - 2 * 3);
  EXPECT_EQ(
      getGlobalOffsetDifference({2, 3, 5}, {7, 11, 13}, {1, 2, 3}, {4, 5, 6}),
      (2 * 7 + 3 * 11 + 5 * 13) - (1 * 4 + 2 * 5 + 3 * 6));
}

TEST_F(AccessPatternCombinationTest, CombinableAccessPatterns) {
  // size(A) == size(B)
  EXPECT_TRUE(checkAccessPaternsCombinable({}, {}, {}, {}, {}, {}, 1));
  EXPECT_TRUE(checkAccessPaternsCombinable({0}, {16}, {1}, {32}, {16}, {1}, 2));
  EXPECT_TRUE(checkAccessPaternsCombinable({0, 0}, {16, 32}, {64, 1}, {0, 32},
                                           {16, 32}, {64, 1}, 4));
  EXPECT_TRUE(checkAccessPaternsCombinable({1, 0}, {16, 32}, {64, 1}, {1, 32},
                                           {16, 32}, {64, 1}, 4));
  EXPECT_TRUE(checkAccessPaternsCombinable({0, 0, 0}, {16, 16, 32}, {32, 64, 1},
                                           {0, 0, 32}, {16, 16, 32},
                                           {32, 64, 1}, 4));
  EXPECT_TRUE(checkAccessPaternsCombinable({0, 2, 0}, {16, 16, 32}, {32, 64, 1},
                                           {0, 2, 32}, {16, 16, 32},
                                           {32, 64, 1}, 4));
  EXPECT_TRUE(checkAccessPaternsCombinable({32, 0}, {64, 64}, {128, 1}, {96, 0},
                                           {32, 64}, {128, 1}, 4));

  // Same access patterns
  EXPECT_TRUE(checkAccessPaternsCombinable({0}, {32}, {1}, {0}, {32}, {1}, 2));
  EXPECT_TRUE(checkAccessPaternsCombinable({0, 0}, {16, 32}, {64, 1}, {0, 0},
                                           {16, 32}, {64, 1}, 4));

  // size(A) > size(B)
  EXPECT_TRUE(checkAccessPaternsCombinable({0, 0, 0}, {2, 16, 32}, {32, 64, 1},
                                           {0, 64}, {16, 32}, {64, 1}, 4));
  EXPECT_TRUE(checkAccessPaternsCombinable({0, 0, 0}, {2, 16, 32}, {32, 64, 1},
                                           {1, 0}, {16, 32}, {64, 1}, 4));
  EXPECT_TRUE(checkAccessPaternsCombinable({0, 0, 32}, {2, 16, 32}, {32, 64, 1},
                                           {0, 96}, {16, 32}, {64, 1}, 4));
  EXPECT_TRUE(checkAccessPaternsCombinable({0, 2, 0}, {2, 16, 32}, {32, 16, 1},
                                           {6, 0}, {16, 32}, {16, 1}, 4));

  // size(A) > size(B) Same access pattern
  EXPECT_TRUE(
      checkAccessPaternsCombinable({0, 0}, {0, 32}, {0, 1}, {0}, {32}, {1}, 2));
  EXPECT_TRUE(
      checkAccessPaternsCombinable({0, 0}, {7, 32}, {0, 1}, {0}, {32}, {1}, 2));
  EXPECT_TRUE(checkAccessPaternsCombinable({1, 0, 0}, {8, 16, 32}, {0, 64, 1},
                                           {0, 0}, {16, 32}, {64, 1}, 4));

  // size(B) > size(A)
  EXPECT_TRUE(checkAccessPaternsCombinable(
      {0, 0}, {16, 32}, {64, 1}, {0, 0, 32}, {2, 16, 32}, {32, 64, 1}, 4));
  EXPECT_TRUE(checkAccessPaternsCombinable({0, 0}, {16, 32}, {16, 1}, {0, 2, 0},
                                           {2, 16, 32}, {32, 16, 1}, 4));
  EXPECT_TRUE(checkAccessPaternsCombinable(
      {0, 32}, {16, 32}, {64, 1}, {0, 0, 64}, {2, 16, 32}, {32, 64, 1}, 4));
  EXPECT_TRUE(checkAccessPaternsCombinable({2, 0}, {16, 32}, {16, 1}, {0, 4, 0},
                                           {2, 16, 32}, {32, 16, 1}, 4));
}

TEST_F(AccessPatternCombinationTest, NonCombinableAccessPatterns) {
  // |size(A) - size(B)| > 1
  EXPECT_FALSE(
      checkAccessPaternsCombinable({}, {}, {}, {0, 0}, {16, 32}, {64, 1}, 3));
  EXPECT_FALSE(checkAccessPaternsCombinable({0}, {32}, {1}, {0, 0, 32},
                                            {2, 16, 32}, {128, 64, 1}, 3));
  EXPECT_FALSE(
      checkAccessPaternsCombinable({0, 0}, {16, 32}, {64, 1}, {}, {}, {}, 3));
  EXPECT_FALSE(checkAccessPaternsCombinable({0, 0, 32}, {2, 16, 32},
                                            {128, 64, 1}, {0}, {32}, {1}, 3));

  // Too few dimensions
  EXPECT_FALSE(
      checkAccessPaternsCombinable({0}, {16}, {1}, {32}, {16}, {1}, 1));
  EXPECT_FALSE(checkAccessPaternsCombinable({0}, {32}, {1}, {0}, {32}, {1}, 1));
  EXPECT_FALSE(checkAccessPaternsCombinable({0, 0}, {16, 32}, {64, 1}, {0, 32},
                                            {16, 32}, {64, 1}, 2));
  EXPECT_FALSE(checkAccessPaternsCombinable({0, 0, 0}, {16, 16, 32},
                                            {32, 64, 1}, {0, 0, 32},
                                            {16, 16, 32}, {32, 64, 1}, 3));

  // size(A) > size(B) Incompatible offset
  EXPECT_FALSE(checkAccessPaternsCombinable({0, 0, 0}, {2, 16, 32}, {32, 64, 1},
                                            {0, 32}, {16, 32}, {64, 1}, 4));
  EXPECT_FALSE(checkAccessPaternsCombinable({0, 0, 0}, {2, 16, 32}, {32, 64, 1},
                                            {0, 128}, {16, 32}, {64, 1}, 4));
  EXPECT_FALSE(checkAccessPaternsCombinable({0, 0, 0}, {2, 16, 32}, {32, 64, 1},
                                            {64, 0}, {16, 32}, {64, 1}, 4));

  // size(A) > size(B) Same access pattern
  EXPECT_FALSE(checkAccessPaternsCombinable({0, 0}, {32, 64}, {128, 1}, {0},
                                            {64}, {1}, 4));
  EXPECT_FALSE(checkAccessPaternsCombinable({1, 0}, {32, 64}, {128, 1}, {0},
                                            {64}, {1}, 4));
  EXPECT_FALSE(checkAccessPaternsCombinable(
      {0, 0, 0}, {32, 64, 128}, {32, 128, 1}, {0, 0}, {64, 128}, {128, 1}, 4));
  EXPECT_FALSE(checkAccessPaternsCombinable(
      {2, 0, 0}, {32, 64, 128}, {32, 128, 1}, {0, 0}, {64, 128}, {128, 1}, 4));

  // size(B) > size(A) Incompatible offset
  EXPECT_FALSE(checkAccessPaternsCombinable(
      {0, 0}, {16, 32}, {64, 1}, {0, 0, 16}, {2, 16, 32}, {32, 64, 1}, 4));
  EXPECT_FALSE(checkAccessPaternsCombinable(
      {0, 0}, {16, 32}, {64, 1}, {0, 0, 96}, {2, 16, 32}, {32, 64, 1}, 4));
  EXPECT_FALSE(checkAccessPaternsCombinable(
      {0, 0}, {16, 32}, {64, 1}, {0, 1, 0}, {2, 16, 32}, {32, 64, 1}, 4));

  // size(B) > size(A) Same access pattern
  EXPECT_FALSE(
      checkAccessPaternsCombinable({0}, {32}, {1}, {0, 0}, {2, 32}, {8, 1}, 4));
  EXPECT_FALSE(
      checkAccessPaternsCombinable({0}, {32}, {1}, {2, 0}, {2, 32}, {8, 1}, 4));

  // size(A) == size(B)
  EXPECT_FALSE(checkAccessPaternsCombinable({32, 0}, {64, 64}, {128, 1},
                                            {32, 0}, {32, 64}, {128, 1}, 4));
  EXPECT_FALSE(checkAccessPaternsCombinable({32, 0}, {32, 64}, {128, 1},
                                            {96, 0}, {64, 64}, {128, 1}, 4));
}

TEST_F(AccessPatternCombinationTest, AnyNbDims) {
  auto exceedsNbDims = [](size_t dims) { return false; };
  EXPECT_TRUE(checkAccessPaternsCombinable({0}, {16}, {1}, {32}, {16}, {1},
                                           exceedsNbDims));
  EXPECT_TRUE(checkAccessPaternsCombinable({0, 0, 0}, {16, 16, 32}, {32, 64, 1},
                                           {0, 0, 32}, {16, 16, 32},
                                           {32, 64, 1}, exceedsNbDims));
}

TEST_F(AccessPatternCombinationTest, NoDims) {
  auto exceedsNbDims = [](size_t dims) { return true; };
  EXPECT_FALSE(checkAccessPaternsCombinable({0}, {16}, {1}, {32}, {16}, {1},
                                            exceedsNbDims));
  EXPECT_FALSE(checkAccessPaternsCombinable(
      {0, 0, 0}, {16, 16, 32}, {32, 64, 1}, {0, 0, 32}, {16, 16, 32},
      {32, 64, 1}, exceedsNbDims));
}

TEST_F(AccessPatternCombinationTest, CombineAccessPatterns) {
  // size(A) == size(B)
  EXPECT_TRUE(checkCombine({0, 0}, {8, 16}, {8, 1}, {0, 32}, {8, 16}, {8, 1},
                           {0, 0, 0}, {2, 8, 16}, {32, 8, 1}, 3));
  EXPECT_TRUE(checkCombine({}, {}, {}, {}, {}, {}, {}, {}, {}, 1));
  EXPECT_TRUE(checkCombine({0}, {16}, {1}, {32}, {16}, {1}, {0, 0}, {2, 16},
                           {32, 1}, 2));
  EXPECT_TRUE(checkCombine({0, 32}, {8, 16}, {8, 1}, {0, 64}, {8, 16}, {8, 1},
                           {0, 0, 32}, {2, 8, 16}, {32, 8, 1}, 3));
  EXPECT_TRUE(checkCombine({1, 32}, {8, 16}, {8, 1}, {1, 64}, {8, 16}, {8, 1},
                           {0, 1, 32}, {2, 8, 16}, {32, 8, 1}, 3));
  EXPECT_TRUE(checkCombine({0, 0}, {8, 16}, {8, 1}, {32, 0}, {8, 16}, {8, 1},
                           {0, 0, 0}, {2, 8, 16}, {256, 8, 1}, 3));
  EXPECT_TRUE(checkCombine({8, 0}, {8, 16}, {8, 1}, {40, 0}, {8, 16}, {8, 1},
                           {0, 8, 0}, {2, 8, 16}, {256, 8, 1}, 3));
  EXPECT_TRUE(checkCombine({0, 0, 0}, {16, 8, 16}, {16, 8, 1}, {0, 0, 32},
                           {16, 8, 16}, {16, 8, 1}, {0, 0, 0, 0},
                           {2, 16, 8, 16}, {32, 16, 8, 1}, 4));
  EXPECT_TRUE(checkCombine({0, 0, 32}, {16, 8, 16}, {16, 8, 1}, {0, 0, 64},
                           {16, 8, 16}, {16, 8, 1}, {0, 0, 0, 32},
                           {2, 16, 8, 16}, {32, 16, 8, 1}, 4));
  EXPECT_TRUE(checkCombine({0, 0, 0}, {16, 8, 16}, {16, 8, 1}, {32, 0, 0},
                           {16, 8, 16}, {16, 8, 1}, {0, 0, 0, 0},
                           {2, 16, 8, 16}, {512, 16, 8, 1}, 4));
  EXPECT_TRUE(checkCombine({8, 0, 0}, {16, 8, 16}, {16, 8, 1}, {40, 0, 0},
                           {16, 8, 16}, {16, 8, 1}, {0, 8, 0, 0},
                           {2, 16, 8, 16}, {512, 16, 8, 1}, 4));
  EXPECT_TRUE(checkCombine({32, 0}, {64, 64}, {128, 1}, {96, 0}, {32, 64},
                           {128, 1}, {32, 0}, {96, 64}, {128, 1}, 4));

  // size(A) == size(B) Same access pattern
  EXPECT_TRUE(
      checkCombine({0}, {32}, {1}, {0}, {32}, {1}, {0, 0}, {2, 32}, {0, 1}, 2));
  EXPECT_TRUE(checkCombine({0, 0}, {16, 32}, {16, 1}, {0, 0}, {16, 32}, {16, 1},
                           {0, 0, 0}, {2, 16, 32}, {0, 16, 1}, 3));

  // size(A) > size(B)
  EXPECT_TRUE(checkCombine({0, 0}, {2, 32}, {64, 1}, {128}, {32}, {1}, {0, 0},
                           {3, 32}, {64, 1}, 3, true));
  EXPECT_TRUE(checkCombine({0, 32}, {3, 32}, {64, 1}, {224}, {32}, {1}, {0, 32},
                           {4, 32}, {64, 1}, 3, true));
  EXPECT_TRUE(checkCombine({0, 0, 0}, {2, 16, 32}, {32, 64, 1}, {0, 64},
                           {16, 32}, {64, 1}, {0, 0, 0}, {3, 16, 32},
                           {32, 64, 1}, 4));
  EXPECT_TRUE(checkCombine({0, 0, 0}, {2, 16, 32}, {32, 64, 1}, {1, 0},
                           {16, 32}, {64, 1}, {0, 0, 0}, {3, 16, 32},
                           {32, 64, 1}, 4));
  EXPECT_TRUE(checkCombine({0, 1, 0}, {2, 16, 32}, {32, 64, 1}, {2, 0},
                           {16, 32}, {64, 1}, {0, 1, 0}, {3, 16, 32},
                           {32, 64, 1}, 4));
  EXPECT_TRUE(checkCombine({0, 1, 32}, {2, 16, 32}, {32, 64, 1}, {2, 32},
                           {16, 32}, {64, 1}, {0, 1, 32}, {3, 16, 32},
                           {32, 64, 1}, 4));

  // size(A) > size(B) Same access pattern
  EXPECT_TRUE(checkCombine({0, 0}, {7, 32}, {0, 1}, {0}, {32}, {1}, {0, 0},
                           {8, 32}, {0, 1}, 3, true));
  EXPECT_TRUE(checkCombine({1, 0}, {7, 32}, {0, 1}, {0}, {32}, {1}, {1, 0},
                           {8, 32}, {0, 1}, 3, true));
  EXPECT_TRUE(
      checkCombine({1, 0}, {0, 32}, {0, 1}, {0}, {32}, {1}, {0}, {32}, {1}, 3));

  // size(B) > size(A)
  EXPECT_TRUE(checkCombine({0}, {32}, {1}, {0, 64}, {2, 32}, {64, 1}, {0, 0},
                           {3, 32}, {64, 1}, 3, true));
  EXPECT_TRUE(checkCombine({32}, {32}, {1}, {0, 96}, {2, 32}, {64, 1}, {0, 32},
                           {3, 32}, {64, 1}, 3, true));
  EXPECT_TRUE(checkCombine({0, 0}, {16, 32}, {16, 1}, {0, 0, 64}, {2, 16, 32},
                           {64, 16, 1}, {0, 0, 0}, {3, 16, 32}, {64, 16, 1},
                           4));
  EXPECT_TRUE(checkCombine({0, 32}, {16, 32}, {16, 1}, {0, 0, 96}, {2, 16, 32},
                           {64, 16, 1}, {0, 0, 32}, {3, 16, 32}, {64, 16, 1},
                           4));
  EXPECT_TRUE(checkCombine({2, 0}, {16, 32}, {16, 1}, {0, 6, 0}, {2, 16, 32},
                           {64, 16, 1}, {0, 2, 0}, {3, 16, 32}, {64, 16, 1},
                           4));
  EXPECT_TRUE(checkCombine({2, 32}, {16, 32}, {16, 1}, {0, 6, 32}, {2, 16, 32},
                           {64, 16, 1}, {0, 2, 32}, {3, 16, 32}, {64, 16, 1},
                           4));
  EXPECT_TRUE(checkCombine({0}, {32}, {1}, {1, 0}, {2, 32}, {64, 1}, {0, 0},
                           {3, 32}, {64, 1}, 3, true));

  // size(B) > size(A) Same access pattern
  EXPECT_TRUE(checkCombine({0}, {32}, {1}, {1, 0}, {3, 32}, {16, 1}, {0, 0},
                           {4, 32}, {16, 1}, 3, true));
  EXPECT_TRUE(checkCombine({0, 0}, {16, 32}, {16, 1}, {1, 0, 0}, {3, 16, 32},
                           {64, 16, 1}, {0, 0, 0}, {4, 16, 32}, {64, 16, 1},
                           3));
  EXPECT_TRUE(checkCombine({}, {}, {}, {}, {}, {}, {}, {}, {}, 100));
  EXPECT_TRUE(checkCombine({0, 0}, {16, 32}, {16, 1}, {0, 32}, {16, 32},
                           {16, 1}, {0, 0, 0}, {2, 16, 32}, {32, 16, 1}, 100));
}

TEST_F(AccessPatternCombinationTest, StrideMatching) {
  EXPECT_TRUE(checkCombine({0, 0}, {8, 16}, {64, 1},      //
                           {8 * 64, 0}, {1, 16}, {1, 1},  //
                           {0, 0}, {9, 16}, {64, 1}, 3));

  // offset A : 4*64
  // offset B : 12*64
  // offset difference: 8*64
  // for A, size[0]*stride[0] is 8*64 (= offset difference)
  // ==> no new dimension required.
  EXPECT_TRUE(checkCombine({4, 0}, {8, 16}, {64, 1},       //
                           {12 * 64, 0}, {1, 16}, {1, 1},  //
                           {4, 0}, {9, 16}, {64, 1}, 3));

  // First access pattern is [10,...19]
  // Second access pattern is [13,...,22].
  //
  // This is
  // for d0 = 0:2
  //   for d1 = 0:10
  //    access (d0+0)*3 + (d1+10)*1
  //
  // from which we one set of possible new params:
  // size: [2, 10] offset: [0, 10] stride:[3, 1]
  EXPECT_TRUE(checkCombine({10}, {10}, {1},          //
                           {2, 5}, {1, 10}, {4, 1},  //
                           {0, 10}, {2, 10}, {3, 1}, 4, true));

  // First access pattern is [10,...,19]
  // Second access pattern is [8,...,17].
  // so they cannot be merged, as the difference between the pointers to the
  // first elements is negative.
  EXPECT_FALSE(checkCombine({10}, {10}, {1},          //
                            {2, 0}, {1, 10}, {4, 1},  //
                            {}, {}, {}, 4, false));

  // First access pattern is [10,...,19]
  // Second access pattern is [20,...,29].
  // so these could in theory be combined into a rank-1 access, but merging
  // dimensions that are not size 1 is not the responsibility of this function.
  EXPECT_TRUE(checkCombine({10}, {10}, {1},          //
                           {4, 4}, {1, 10}, {4, 1},  //
                           {0, 10}, {2, 10}, {10, 1}, 4, true));
}

TEST_F(AccessPatternCombinationTest, FailCombineAccessPatterns) {
  // |size(A) - size(B)| > 1
  EXPECT_FALSE(checkCombine({}, {}, {}, {0, 0}, {16, 32}, {16, 1}, {}, {}, {},
                            3, false));
  EXPECT_FALSE(checkCombine({0, 0}, {16, 32}, {16, 1}, {}, {}, {}, {}, {}, {},
                            3, false));
  // Too few dimensions
  EXPECT_FALSE(checkCombine({0, 0}, {16, 32}, {16, 1}, {0, 32}, {16, 32},
                            {16, 1}, {}, {}, {}, 2, false));

  // size(A) > size(B) Incompatible offset
  EXPECT_FALSE(checkCombine({0, 0}, {2, 32}, {64, 1}, {96}, {32}, {1}, {0, 0},
                            {3, 32}, {64, 1}, 3, false));
  EXPECT_FALSE(checkCombine({0, 0}, {2, 32}, {64, 1}, {256}, {32}, {1}, {0, 0},
                            {3, 32}, {64, 1}, 3, false));
  // size(B) > size(A) Same access pattern
  EXPECT_FALSE(checkCombine({0, 0}, {16, 32}, {16, 1}, {0, 0, 0}, {3, 16, 32},
                            {64, 16, 1}, {0, 0, 0}, {4, 16, 32}, {64, 16, 1}, 3,
                            false));
  EXPECT_FALSE(checkCombine({0, 0}, {16, 32}, {16, 1}, {2, 0, 0}, {3, 16, 32},
                            {64, 16, 1}, {0, 0, 0}, {4, 16, 32}, {64, 16, 1}, 3,
                            false));

  // size(B) > size(A) Incompatible offset
  EXPECT_FALSE(checkCombine({0}, {32}, {1}, {0, 32}, {2, 32}, {64, 1}, {0, 0},
                            {3, 32}, {64, 1}, 3, false));
  EXPECT_FALSE(checkCombine({0}, {32}, {1}, {0, 96}, {2, 32}, {64, 1}, {0, 0},
                            {3, 32}, {64, 1}, 3, false));

  // size(A) == size(B) Incompatible offset
  EXPECT_FALSE(checkCombine({32, 0}, {32, 64}, {128, 1}, {96, 0}, {64, 64},
                            {128, 1}, {32, 0}, {96, 64}, {128, 1}, 4, false));
  EXPECT_FALSE(
      checkCombine({0}, {16}, {1}, {32}, {16}, {1}, {}, {}, {}, 0, false));
  EXPECT_FALSE(checkCombine({0, 0}, {16, 32}, {16, 1}, {0, 32}, {16, 32},
                            {16, 1}, {0, 0, 0}, {2, 16, 32}, {32, 16, 1}, 0,
                            false));
}

//===----------------------------------------------------------------------===//
// Fold and Expand Tests
//===----------------------------------------------------------------------===//

class FoldAndExpandTest : public ::testing::Test {
 protected:
  FoldAndExpandTest() : rewriter(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<arith::ArithDialect>();
  }

  SmallVector<OpFoldResult> toOpFoldResults(SmallVector<int64_t> values) {
    return llvm::map_to_vector(values, [&](int64_t v) -> OpFoldResult {
      return getAsIndexOpFoldResult(&context, v);
    });
  }

  void checkExpandDimIntoLinearDims(
      SmallVector<int64_t> offsets, SmallVector<int64_t> sizes,
      SmallVector<int64_t> strides, SmallVector<int64_t> maxSizes,
      SmallVector<int64_t> expectedOffsets, SmallVector<int64_t> expectedSizes,
      SmallVector<int64_t> expectedStrides, bool shouldSucceed = true) {
    SmallVector<OpFoldResult> offsetsValues = toOpFoldResults(offsets);
    SmallVector<OpFoldResult> sizesValues = toOpFoldResults(sizes);
    SmallVector<OpFoldResult> stridesValues = toOpFoldResults(strides);

    SmallVector<OpFoldResult> expectedOffsetsValues =
        toOpFoldResults(expectedOffsets);
    SmallVector<OpFoldResult> expectedSizesValues =
        toOpFoldResults(expectedSizes);
    SmallVector<OpFoldResult> expectedStridesValues =
        toOpFoldResults(expectedStrides);
    SmallVector<OpFoldResult> newOffsets;
    SmallVector<OpFoldResult> newSizes;
    SmallVector<OpFoldResult> newStrides;
    if (shouldSucceed) {
      EXPECT_TRUE(succeeded(expandLargeDimIntoLinearDims(
          &context, offsetsValues, sizesValues, stridesValues, newOffsets,
          newSizes, newStrides, maxSizes)));
    } else {
      EXPECT_TRUE(failed(expandLargeDimIntoLinearDims(
          &context, offsetsValues, sizesValues, stridesValues, newOffsets,
          newSizes, newStrides, maxSizes)));
    }
    EXPECT_EQ(newOffsets, expectedOffsetsValues);
    EXPECT_EQ(newSizes, expectedSizesValues);
    EXPECT_EQ(newStrides, expectedStridesValues);
  }

  void checkFoldLinearDims(
      SmallVector<int64_t> offsets, SmallVector<int64_t> sizes,
      SmallVector<int64_t> strides, SmallVector<int64_t> maxSizes,
      SmallVector<int64_t> expectedOffsets, SmallVector<int64_t> expectedSizes,
      SmallVector<int64_t> expectedStrides, bool shouldSucceed = true) {
    SmallVector<OpFoldResult> offsetsValues = toOpFoldResults(offsets);
    SmallVector<OpFoldResult> sizesValues = toOpFoldResults(sizes);
    SmallVector<OpFoldResult> stridesValues = toOpFoldResults(strides);

    SmallVector<OpFoldResult> expectedOffsetsValues =
        toOpFoldResults(expectedOffsets);
    SmallVector<OpFoldResult> expectedSizesValues =
        toOpFoldResults(expectedSizes);
    SmallVector<OpFoldResult> expectedStridesValues =
        toOpFoldResults(expectedStrides);
    SmallVector<OpFoldResult> newOffsets;
    SmallVector<OpFoldResult> newSizes;
    SmallVector<OpFoldResult> newStrides;
    auto isValidSize = [&](size_t idxFromEnd, int64_t size) -> bool {
      if (maxSizes.empty()) return true;
      return idxFromEnd < maxSizes.size() &&
             size <= maxSizes[maxSizes.size() - idxFromEnd - 1];
    };
    if (shouldSucceed) {
      EXPECT_TRUE(succeeded(foldLinearDims(&context, offsetsValues, sizesValues,
                                           stridesValues, newOffsets, newSizes,
                                           newStrides, isValidSize)));
    } else {
      EXPECT_TRUE(failed(foldLinearDims(&context, offsetsValues, sizesValues,
                                        stridesValues, newOffsets, newSizes,
                                        newStrides, isValidSize)));
    }
    EXPECT_EQ(newOffsets, expectedOffsetsValues);
    EXPECT_EQ(newSizes, expectedSizesValues);
    EXPECT_EQ(newStrides, expectedStridesValues);
  }

  bool checkFoldUnitDims(SmallVector<int64_t> offsets,
                         SmallVector<int64_t> sizes,
                         SmallVector<int64_t> strides,
                         SmallVector<int64_t> expectedOffsets,
                         SmallVector<int64_t> expectedSizes,
                         SmallVector<int64_t> expectedStrides) {
    SmallVector<OpFoldResult> offsetsValues = toOpFoldResults(offsets);
    SmallVector<OpFoldResult> sizesValues = toOpFoldResults(sizes);
    SmallVector<OpFoldResult> stridesValues = toOpFoldResults(strides);

    auto folded =
        foldUnitDims(&context, offsetsValues, sizesValues, stridesValues);

    EXPECT_EQ(fromOpFoldResults(offsetsValues), expectedOffsets);
    EXPECT_EQ(fromOpFoldResults(sizesValues), expectedSizes);
    EXPECT_EQ(fromOpFoldResults(stridesValues), expectedStrides);

    return succeeded(folded);
  }

  bool checkNotFoldUnitDims(SmallVector<int64_t> offsets,
                            SmallVector<int64_t> sizes,
                            SmallVector<int64_t> strides) {
    return checkFoldUnitDims(offsets, sizes, strides, offsets, sizes, strides);
  }

  void checkFoldRepetitionCount(
      const SmallVector<int64_t> sizes, const SmallVector<int64_t> strides,
      const SmallVector<int64_t> expectedSizes,
      const SmallVector<int64_t> expectedStrides,
      std::optional<int64_t> maybeRepetitionCount = std::nullopt,
      bool shouldSucceed = true) {
    SmallVector<OpFoldResult> sizesValues = toOpFoldResults(sizes);
    SmallVector<OpFoldResult> stridesValues = toOpFoldResults(strides);
    SmallVector<OpFoldResult> expectedSizesValues =
        toOpFoldResults(expectedSizes);
    SmallVector<OpFoldResult> expectedStridesValues =
        toOpFoldResults(expectedStrides);
    if (shouldSucceed) {
      EXPECT_TRUE(succeeded(foldRepetitionCount(
          &context, sizesValues, stridesValues, maybeRepetitionCount)));
      EXPECT_EQ(sizesValues, expectedSizesValues);
      EXPECT_EQ(stridesValues, expectedStridesValues);
    } else {
      EXPECT_TRUE(failed(foldRepetitionCount(
          &context, sizesValues, stridesValues, maybeRepetitionCount)));
    }
  }

  MLIRContext context;
  IRRewriter rewriter;
  Location loc;
};

TEST_F(FoldAndExpandTest, NoLinearDimsFold) {
  checkFoldLinearDims({}, {}, {}, {}, {}, {}, {}, false);
  checkFoldLinearDims({0}, {8}, {1}, {}, {0}, {8}, {1}, false);
  checkFoldLinearDims({0, 0}, {16, 8}, {16, 1}, {}, {0, 0}, {16, 8}, {16, 1},
                      false);
}

TEST_F(FoldAndExpandTest, FoldLinearDims) {
  checkFoldLinearDims({0, 0}, {16, 8}, {8, 1}, {}, {0}, {128}, {1}, true);
  checkFoldLinearDims({0, 8}, {16, 8}, {8, 1}, {}, {8}, {128}, {1}, true);
  checkFoldLinearDims({0, 0, 0}, {8, 16, 8}, {128, 8, 1}, {}, {0}, {1024}, {1},
                      true);
  checkFoldLinearDims({0, 0, 0, 0}, {4, 8, 16, 8}, {1024, 128, 8, 1}, {}, {0},
                      {4096}, {1}, true);
  checkFoldLinearDims({5, 3, 8, 1}, {4, 8, 16, 8}, {1024, 128, 8, 1}, {},
                      {5569}, {4096}, {1}, true);
}

TEST_F(FoldAndExpandTest, FoldLinearDimsWithMax) {
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
  checkFoldLinearDims({4, 0, 8, 0}, {4, 8, 16, 8}, {1024, 128, 8, 1},
                      {511, 511, 511, 511}, {32, 64}, {32, 128}, {128, 1},
                      true);
}

TEST_F(FoldAndExpandTest, NoUnitDimsFold) {
  EXPECT_FALSE(checkNotFoldUnitDims({}, {}, {}));
  EXPECT_FALSE(checkNotFoldUnitDims({0}, {8}, {1}));
  EXPECT_FALSE(checkNotFoldUnitDims({0, 0}, {16, 8}, {16, 1}));
  EXPECT_FALSE(checkNotFoldUnitDims({2}, {1}, {1}));
}

TEST_F(FoldAndExpandTest, UnitDimsFullFold) {
  EXPECT_TRUE(checkFoldUnitDims({0}, {1}, {32}, {}, {}, {}));
  EXPECT_TRUE(checkFoldUnitDims({0, 0, 0}, {32, 1, 8}, {32, 1024, 1}, {0, 0},
                                {32, 8}, {32, 1}));
  EXPECT_TRUE(checkFoldUnitDims({0, 0, 0, 0}, {1, 32, 1, 8},
                                {1024, 32, 1024, 1}, {0, 0}, {32, 8}, {32, 1}));
}

TEST_F(FoldAndExpandTest, UnitDimsMerge) {
  EXPECT_TRUE(checkFoldUnitDims({1, 1}, {1, 1}, {32, 32}, {1}, {1}, {64}));
  EXPECT_TRUE(checkFoldUnitDims({1, 2}, {1, 1}, {32, 32}, {1}, {1}, {96}));
  EXPECT_TRUE(checkFoldUnitDims({2, 1}, {1, 1}, {32, 32}, {1}, {1}, {96}));
  EXPECT_TRUE(checkFoldUnitDims({1, 0, 1, 0}, {1, 32, 1, 8},
                                {1024, 32, 1024, 1}, {64, 0}, {32, 8},
                                {32, 1}));
  EXPECT_TRUE(checkFoldUnitDims({1, 0, 2, 0}, {1, 32, 1, 8},
                                {1024, 32, 1024, 1}, {96, 0}, {32, 8},
                                {32, 1}));
  EXPECT_TRUE(checkFoldUnitDims({2, 0, 1, 0}, {1, 32, 1, 8},
                                {1024, 32, 1024, 1}, {96, 0}, {32, 8},
                                {32, 1}));
  EXPECT_TRUE(checkFoldUnitDims({0, 0, 1, 0}, {2, 32, 1, 8}, {0, 32, 1024, 1},
                                {0, 32, 0}, {2, 32, 8}, {0, 32, 1}));
  EXPECT_TRUE(
      checkFoldUnitDims({2, 2, 15}, {1, 1, 10}, {4, 6, 10}, {17}, {10}, {10}));
  EXPECT_TRUE(checkFoldUnitDims({3, 1, 15}, {1, 1, 10}, {4, 6, 10}, {1, 15},
                                {1, 10}, {18, 10}));
}

TEST_F(FoldAndExpandTest, UnitDimsFoldAndMerge) {
  EXPECT_TRUE(
      checkFoldUnitDims({1, 0, 1}, {1, 1, 1}, {32, 1024, 32}, {1}, {1}, {64}));
  EXPECT_TRUE(
      checkFoldUnitDims({1, 0, 1}, {1, 1, 1}, {32, 32, 32}, {1}, {1}, {64}));
  EXPECT_TRUE(checkFoldUnitDims({1, 0, 2, 0}, {1, 1, 1, 1}, {32, 32, 32, 32},
                                {1}, {1}, {96}));
  EXPECT_TRUE(checkFoldUnitDims({1, 0, 1, 0}, {1, 1, 1, 8}, {1024, 32, 1024, 1},
                                {2048}, {8}, {1}));
  EXPECT_TRUE(checkFoldUnitDims({0, 0, 1, 0}, {1, 32, 1, 8}, {0, 32, 1024, 1},
                                {32, 0}, {32, 8}, {32, 1}));
}

TEST_F(FoldAndExpandTest, FoldRepetitionCount) {
  checkFoldRepetitionCount({2}, {0}, {1}, {0});
  checkFoldRepetitionCount({2}, {0}, {1}, {0}, 2);
  checkFoldRepetitionCount({4, 3}, {0, 1}, {2, 3}, {0, 1}, 2);
  checkFoldRepetitionCount({4, 3}, {0, 0}, {2, 3}, {0, 0}, 2);
  checkFoldRepetitionCount({4, 3}, {0, 0}, {1, 1}, {0, 0});
  checkFoldRepetitionCount({4, 3}, {0, 0}, {1, 1}, {0, 0}, 12);
  checkFoldRepetitionCount({4, 6, 4}, {0, 0, 1}, {1, 1, 4}, {0, 0, 1});
  checkFoldRepetitionCount({4, 6, 4}, {0, 0, 1}, {2, 6, 4}, {0, 0, 1}, 2);
  checkFoldRepetitionCount({4, 6, 4}, {0, 0, 1}, {1, 6, 4}, {0, 0, 1}, 4);
  checkFoldRepetitionCount({4, 6, 4}, {0, 0, 1}, {1, 3, 4}, {0, 0, 1}, 8);
  checkFoldRepetitionCount({4, 6, 4}, {0, 0, 1}, {1, 2, 4}, {0, 0, 1}, 12);
  checkFoldRepetitionCount({4, 6, 4}, {0, 0, 1}, {1, 1, 4}, {0, 0, 1}, 24);
}

TEST_F(FoldAndExpandTest, NoFoldRepetitionCount) {
  checkFoldRepetitionCount({}, {}, {}, {});
  checkFoldRepetitionCount({}, {}, {}, {}, 1);
  checkFoldRepetitionCount({2}, {1}, {2}, {1});
  checkFoldRepetitionCount({2}, {1}, {2}, {1}, 1);
  checkFoldRepetitionCount({4, 2}, {8, 1}, {4, 2}, {8, 1});
  checkFoldRepetitionCount({4, 2}, {8, 1}, {4, 2}, {8, 1}, 1);
  checkFoldRepetitionCount({4, 2}, {0, 1}, {4, 2}, {0, 1}, 1);
}

TEST_F(FoldAndExpandTest, FoldRepetitionCountFail) {
  checkFoldRepetitionCount({}, {}, {}, {}, 2, false);
  checkFoldRepetitionCount({1}, {1}, {}, {}, 2, false);
  checkFoldRepetitionCount({3}, {0}, {}, {}, 2, false);
  checkFoldRepetitionCount({4, 3}, {0, 0}, {1, 1}, {0, 0}, 8, false);
  checkFoldRepetitionCount({4, 3}, {0, 0}, {1, 1}, {0, 0}, 24, false);
  checkFoldRepetitionCount({4, 6, 4}, {0, 0, 1}, {1, 2, 4}, {0, 0, 1}, 7,
                           false);
  checkFoldRepetitionCount({4, 6, 4}, {0, 0, 1}, {1, 2, 4}, {0, 0, 1}, 16,
                           false);
  checkFoldRepetitionCount({4, 6, 4}, {0, 0, 1}, {1, 2, 4}, {0, 0, 1}, 48,
                           false);
}

TEST_F(FoldAndExpandTest, NoDimsExpand) {
  checkExpandDimIntoLinearDims({}, {}, {}, {}, {}, {}, {}, false);
  checkExpandDimIntoLinearDims({0}, {1}, {1}, {1}, {0}, {1}, {1}, false);
  checkExpandDimIntoLinearDims({0}, {8}, {1}, {8}, {0}, {8}, {1}, false);
  checkExpandDimIntoLinearDims({0}, {11}, {1}, {7}, {0}, {11}, {1}, false);
  checkExpandDimIntoLinearDims({0, 8}, {16, 8}, {16, 1}, {16, 8}, {0, 8},
                               {16, 8}, {16, 1}, false);
}

TEST_F(FoldAndExpandTest, ExpandLargeDim) {
  checkExpandDimIntoLinearDims({0}, {8}, {1}, {7}, {0, 0}, {2, 4}, {4, 1},
                               true);
  checkExpandDimIntoLinearDims({0}, {8}, {1}, {5}, {0, 0}, {2, 4}, {4, 1},
                               true);
  checkExpandDimIntoLinearDims({0}, {8}, {1}, {4}, {0, 0}, {2, 4}, {4, 1},
                               true);
  checkExpandDimIntoLinearDims({3}, {8}, {1}, {3}, {0, 0, 3}, {2, 2, 2},
                               {4, 2, 1}, true);
  checkExpandDimIntoLinearDims({0, 3}, {2, 8}, {3, 1}, {4, 4}, {0, 0, 3},
                               {2, 2, 4}, {3, 4, 1}, true);
  checkExpandDimIntoLinearDims({0, 3}, {6, 8}, {3, 1}, {3, 4}, {0, 0, 0, 3},
                               {2, 3, 2, 4}, {9, 3, 4, 1}, true);
  checkExpandDimIntoLinearDims({5, 3}, {6, 8}, {3, 1}, {3, 2}, {0, 5, 0, 0, 3},
                               {2, 3, 2, 2, 2}, {9, 3, 4, 2, 1}, true);
}

//===----------------------------------------------------------------------===//
// DmaDimConfig Tests
//===----------------------------------------------------------------------===//

class DmaDimConfigTest : public testing::TestWithParam<AMDAIEDevice> {
 protected:
  DmaDimConfigTest() : deviceModel(getDeviceModel(GetParam())) {}
  AMDAIEDeviceModel deviceModel;
};

TEST_P(DmaDimConfigTest, ShimTileSizes) {
  DmaDimConfig config(deviceModel, 0);
  SmallVector<int64_t> maxSizes = config.getMaxSizes();
  SmallVector<int64_t> expectedMaxSizes = {
      63, std::numeric_limits<int64_t>::max(), 1023, 1023};
  EXPECT_EQ(maxSizes, expectedMaxSizes);
  EXPECT_EQ(config.getMaxSizes(1),
            SmallVector<int64_t>{std::numeric_limits<int64_t>::max()});
  SmallVector<int64_t> expectedMaxSizes2 = {std::numeric_limits<int64_t>::max(),
                                            1023};
  EXPECT_EQ(config.getMaxSizes(2), expectedMaxSizes2);
  SmallVector<int64_t> expectedMaxSizes3 = {std::numeric_limits<int64_t>::max(),
                                            1023, 1023};
  EXPECT_EQ(config.getMaxSizes(3), expectedMaxSizes3);
  EXPECT_EQ(config.getMaxSizes(4), expectedMaxSizes);
  SmallVector<int64_t> expectedMaxSizes5 = {
      0, 63, std::numeric_limits<int64_t>::max(), 1023, 1023};
  EXPECT_EQ(config.getMaxSizes(5), expectedMaxSizes5);
  SmallVector<int64_t> expectedMaxSizes6 = {
      0, 0, 63, std::numeric_limits<int64_t>::max(), 1023, 1023};
  EXPECT_EQ(config.getMaxSizes(6), expectedMaxSizes6);
}

TEST_P(DmaDimConfigTest, ShimTileStrides) {
  DmaDimConfig config(deviceModel, 0);
  SmallVector<int64_t> maxStrides = config.getMaxStrides();
  SmallVector<int64_t> expectedMaxStrides(4, 1 << 20);
  EXPECT_EQ(maxStrides, expectedMaxStrides);
  EXPECT_EQ(config.getMaxStrides(1), SmallVector<int64_t>{1 << 20});
  SmallVector<int64_t> expectedMaxStrides2 = {1 << 20, 1 << 20};
  EXPECT_EQ(config.getMaxStrides(2), expectedMaxStrides2);
  SmallVector<int64_t> expectedMaxStrides3 = {1 << 20, 1 << 20, 1 << 20};
  EXPECT_EQ(config.getMaxStrides(3), expectedMaxStrides3);
  SmallVector<int64_t> expectedMaxStrides4 = {1 << 20, 1 << 20, 1 << 20,
                                              1 << 20};
  EXPECT_EQ(config.getMaxStrides(4), expectedMaxStrides4);
  SmallVector<int64_t> expectedMaxStrides5 = {0, 1 << 20, 1 << 20, 1 << 20,
                                              1 << 20};
  EXPECT_EQ(config.getMaxStrides(5), expectedMaxStrides5);
  SmallVector<int64_t> expectedMaxStrides6 = {0,       0,       1 << 20,
                                              1 << 20, 1 << 20, 1 << 20};
  EXPECT_EQ(config.getMaxStrides(6), expectedMaxStrides6);
}

TEST_P(DmaDimConfigTest, MemTileSizes) {
  DmaDimConfig config(deviceModel, 1);
  SmallVector<int64_t> maxSizes = config.getMaxSizes();
  SmallVector<int64_t> expectedMaxSizes = {std::numeric_limits<int64_t>::max(),
                                           1023, 1023, 1023};
  EXPECT_EQ(maxSizes, expectedMaxSizes);
  EXPECT_EQ(config.getMaxSizes(1),
            SmallVector<int64_t>{std::numeric_limits<int64_t>::max()});
  SmallVector<int64_t> expectedMaxSizes2 = {std::numeric_limits<int64_t>::max(),
                                            1023};
  EXPECT_EQ(config.getMaxSizes(2), expectedMaxSizes2);
  SmallVector<int64_t> expectedMaxSizes3 = {std::numeric_limits<int64_t>::max(),
                                            1023, 1023};
  EXPECT_EQ(config.getMaxSizes(3), expectedMaxSizes3);
  EXPECT_EQ(config.getMaxSizes(4), expectedMaxSizes);
  SmallVector<int64_t> expectedMaxSizes5 = {
      0, std::numeric_limits<int64_t>::max(), 1023, 1023, 1023};
  EXPECT_EQ(config.getMaxSizes(5), expectedMaxSizes5);
  SmallVector<int64_t> expectedMaxSizes6 = {
      0, 0, std::numeric_limits<int64_t>::max(), 1023, 1023, 1023};
  EXPECT_EQ(config.getMaxSizes(6), expectedMaxSizes6);
}

TEST_P(DmaDimConfigTest, MemTileStrides) {
  DmaDimConfig config(deviceModel, 1);
  SmallVector<int64_t> maxStrides = config.getMaxStrides();
  SmallVector<int64_t> expectedMaxStrides(4, 1 << 17);
  EXPECT_EQ(maxStrides, expectedMaxStrides);
  EXPECT_EQ(config.getMaxStrides(1), SmallVector<int64_t>{1 << 17});
  SmallVector<int64_t> expectedMaxStrides2(2, 1 << 17);
  EXPECT_EQ(config.getMaxStrides(2), expectedMaxStrides2);
  SmallVector<int64_t> expectedMaxStrides3(3, 1 << 17);
  EXPECT_EQ(config.getMaxStrides(3), expectedMaxStrides3);
  EXPECT_EQ(config.getMaxStrides(4), expectedMaxStrides);
  SmallVector<int64_t> expectedMaxStrides5 = {0, 1 << 17, 1 << 17, 1 << 17,
                                              1 << 17};
  EXPECT_EQ(config.getMaxStrides(5), expectedMaxStrides5);
  SmallVector<int64_t> expectedMaxStrides6 = {0,       0,       1 << 17,
                                              1 << 17, 1 << 17, 1 << 17};
  EXPECT_EQ(config.getMaxStrides(6), expectedMaxStrides6);
}

TEST_P(DmaDimConfigTest, CoreTileSizes) {
  DmaDimConfig config(deviceModel, 2);
  SmallVector<int64_t> maxSizes = config.getMaxSizes();
  SmallVector<int64_t> expectedMaxSizes = {std::numeric_limits<int64_t>::max(),
                                           255, 255};
  EXPECT_EQ(maxSizes, expectedMaxSizes);
  EXPECT_EQ(config.getMaxSizes(1),
            SmallVector<int64_t>{std::numeric_limits<int64_t>::max()});
  SmallVector<int64_t> expectedMaxSizes2 = {std::numeric_limits<int64_t>::max(),
                                            255};
  EXPECT_EQ(config.getMaxSizes(2), expectedMaxSizes2);
  EXPECT_EQ(config.getMaxSizes(3), expectedMaxSizes);
  SmallVector<int64_t> expectedMaxSizes4 = {
      0, std::numeric_limits<int64_t>::max(), 255, 255};
  EXPECT_EQ(config.getMaxSizes(4), expectedMaxSizes4);
  SmallVector<int64_t> expectedMaxSizes5 = {
      0, 0, std::numeric_limits<int64_t>::max(), 255, 255};
  EXPECT_EQ(config.getMaxSizes(5), expectedMaxSizes5);
}

TEST_P(DmaDimConfigTest, CoreTileStrides) {
  DmaDimConfig config(deviceModel, 2);
  SmallVector<int64_t> maxStrides = config.getMaxStrides();
  SmallVector<int64_t> expectedMaxStrides(3, 1 << 13);
  EXPECT_EQ(maxStrides, expectedMaxStrides);
  EXPECT_EQ(config.getMaxStrides(1), SmallVector<int64_t>{1 << 13});
  SmallVector<int64_t> expectedMaxStrides2(2, 1 << 13);
  EXPECT_EQ(config.getMaxStrides(2), expectedMaxStrides2);
  EXPECT_EQ(config.getMaxStrides(3), expectedMaxStrides);
  SmallVector<int64_t> expectedMaxStrides4 = {0, 1 << 13, 1 << 13, 1 << 13};
  EXPECT_EQ(config.getMaxStrides(4), expectedMaxStrides4);
  SmallVector<int64_t> expectedMaxStrides5 = {0, 0, 1 << 13, 1 << 13, 1 << 13};
  EXPECT_EQ(config.getMaxStrides(5), expectedMaxStrides5);
}

TEST_P(DmaDimConfigTest, CircularShimTileSizes) {
  CircularDmaDimConfig config(deviceModel, 0);
  SmallVector<int64_t> maxSizes = config.getMaxSizes();
  SmallVector<int64_t> expectedMaxSizes = {1023, 1023, 1023, 1023};
  EXPECT_EQ(maxSizes, expectedMaxSizes);
  EXPECT_EQ(config.getMaxSizes(1), SmallVector<int64_t>{1023});
  SmallVector<int64_t> expectedMaxSizes2 = {1023, 1023};
  EXPECT_EQ(config.getMaxSizes(2), expectedMaxSizes2);
  SmallVector<int64_t> expectedMaxSizes3 = {1023, 1023, 1023};
  EXPECT_EQ(config.getMaxSizes(3), expectedMaxSizes3);
  EXPECT_EQ(config.getMaxSizes(4), expectedMaxSizes);
  SmallVector<int64_t> expectedMaxSizes5 = {1023, 1023, 1023, 1023, 1023};
  EXPECT_EQ(config.getMaxSizes(5), expectedMaxSizes5);
}

TEST_P(DmaDimConfigTest, CircularShimTileStrides) {
  CircularDmaDimConfig config(deviceModel, 0);
  SmallVector<int64_t> maxStrides = config.getMaxStrides();
  SmallVector<int64_t> expectedMaxStrides = {
      std::numeric_limits<int64_t>::max(), 1 << 20, 1 << 20, 1 << 20};
  EXPECT_EQ(maxStrides, expectedMaxStrides);
  EXPECT_EQ(config.getMaxStrides(1), SmallVector<int64_t>{1 << 20});
  SmallVector<int64_t> expectedMaxStrides2(2, 1 << 20);
  EXPECT_EQ(config.getMaxStrides(2), expectedMaxStrides2);
  SmallVector<int64_t> expectedMaxStrides3(3, 1 << 20);
  EXPECT_EQ(config.getMaxStrides(3), expectedMaxStrides3);
  EXPECT_EQ(config.getMaxStrides(4), expectedMaxStrides);
  SmallVector<int64_t> expectedMaxStrides5 = {
      std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(),
      1 << 20, 1 << 20, 1 << 20};
  EXPECT_EQ(config.getMaxStrides(5), expectedMaxStrides5);
}

TEST_P(DmaDimConfigTest, CircularMemTileSizes) {
  CircularDmaDimConfig config(deviceModel, 1);
  SmallVector<int64_t> maxSizes = config.getMaxSizes();
  SmallVector<int64_t> expectedMaxSizes = {1023, 1023, 1023, 1023};
  EXPECT_EQ(maxSizes, expectedMaxSizes);
  EXPECT_EQ(config.getMaxSizes(1), SmallVector<int64_t>{1023});
  SmallVector<int64_t> expectedMaxSizes5 = {1023, 1023, 1023, 1023, 1023};
  EXPECT_EQ(config.getMaxSizes(5), expectedMaxSizes5);
}

TEST_P(DmaDimConfigTest, CircularMemTileStrides) {
  CircularDmaDimConfig config(deviceModel, 1);
  SmallVector<int64_t> maxStrides = config.getMaxStrides();
  SmallVector<int64_t> expectedMaxStrides(4, 1 << 17);
  EXPECT_EQ(maxStrides, expectedMaxStrides);
  EXPECT_EQ(config.getMaxStrides(1), SmallVector<int64_t>{1 << 17});
  SmallVector<int64_t> expectedMaxStrides2(2, 1 << 17);
  EXPECT_EQ(config.getMaxStrides(2), expectedMaxStrides2);
  SmallVector<int64_t> expectedMaxStrides3(3, 1 << 17);
  EXPECT_EQ(config.getMaxStrides(3), expectedMaxStrides3);
  EXPECT_EQ(config.getMaxStrides(4), expectedMaxStrides);
  SmallVector<int64_t> expectedMaxStrides5 = {
      std::numeric_limits<int64_t>::max(), 1 << 17, 1 << 17, 1 << 17, 1 << 17};
  EXPECT_EQ(config.getMaxStrides(5), expectedMaxStrides5);
  SmallVector<int64_t> expectedMaxStrides6 = {
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max(),
      1 << 17,
      1 << 17,
      1 << 17,
      1 << 17};
  EXPECT_EQ(config.getMaxStrides(6), expectedMaxStrides6);
}

TEST_P(DmaDimConfigTest, CircularCoreTileSizes) {
  CircularDmaDimConfig config(deviceModel, 2);
  SmallVector<int64_t> maxSizes = config.getMaxSizes();
  SmallVector<int64_t> expectedMaxSizes = {255, 255, 255};
  EXPECT_EQ(maxSizes, expectedMaxSizes);
  EXPECT_EQ(config.getMaxSizes(1), SmallVector<int64_t>{255});
  SmallVector<int64_t> expectedMaxSizes5 = {255, 255, 255, 255, 255};
  EXPECT_EQ(config.getMaxSizes(5), expectedMaxSizes5);
}

TEST_P(DmaDimConfigTest, CircularCoreTileStrides) {
  CircularDmaDimConfig config(deviceModel, 2);
  SmallVector<int64_t> maxStrides = config.getMaxStrides();
  SmallVector<int64_t> expectedMaxStrides(3, 1 << 13);
  EXPECT_EQ(maxStrides, expectedMaxStrides);
  EXPECT_EQ(config.getMaxStrides(1), SmallVector<int64_t>{1 << 13});
  SmallVector<int64_t> expectedMaxStrides2(2, 1 << 13);
  EXPECT_EQ(config.getMaxStrides(2), expectedMaxStrides2);
  EXPECT_EQ(config.getMaxStrides(3), expectedMaxStrides);
  SmallVector<int64_t> expectedMaxStrides4 = {
      std::numeric_limits<int64_t>::max(), 1 << 13, 1 << 13, 1 << 13};
  EXPECT_EQ(config.getMaxStrides(4), expectedMaxStrides4);
  SmallVector<int64_t> expectedMaxStrides5 = {
      std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max(),
      1 << 13, 1 << 13, 1 << 13};
  EXPECT_EQ(config.getMaxStrides(5), expectedMaxStrides5);
}

INSTANTIATE_TEST_SUITE_P(
    Devices, DmaDimConfigTest,
    testing::Values(AMDAIEDevice::npu1, AMDAIEDevice::npu1_1col,
                    AMDAIEDevice::npu1_2col, AMDAIEDevice::npu1_3col,
                    AMDAIEDevice::npu1_4col, AMDAIEDevice::npu4));

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
