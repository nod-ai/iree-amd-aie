// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gtest/gtest.h"
#include "iree-amd-aie/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"

namespace {

using namespace mlir::iree_compiler::AMDAIE;

TEST(GetAccessRangeExtentTest, PositiveSizesAndStrides) {
  EXPECT_EQ(getAccessRangeExtent(/* sizes = */ {}, /* strides = */ {}), 0);
  EXPECT_EQ(getAccessRangeExtent({1}, {1}), 1);
  EXPECT_EQ(getAccessRangeExtent({2}, {1}), 2);
  EXPECT_EQ(getAccessRangeExtent({1}, {2}), 1);
  EXPECT_EQ(getAccessRangeExtent({2, 2}, {4, 1}), 6);
  EXPECT_EQ(getAccessRangeExtent({8, 2}, {4, 1}), 30);
  EXPECT_EQ(getAccessRangeExtent({64, 2, 8, 2}, {3, 16, 4, 1}), 235);
  EXPECT_EQ(getAccessRangeExtent({2, 2}, {4, 2}), 7);
  EXPECT_EQ(getAccessRangeExtent({3, 3}, {4, 4}), 17);
}

TEST(GetAccessRangeExtentTest, NegativeSize) {
  EXPECT_EQ(getAccessRangeExtent(/* sizes = */ {-1}, /* strides = */ {1}),
            std::nullopt);
  EXPECT_EQ(getAccessRangeExtent({2, -4, 1}, {16, 4, 1}), std::nullopt);
}

TEST(GetAccessRangeExtentTest, ZeroSize) {
  EXPECT_EQ(getAccessRangeExtent(/* sizes = */ {0}, /* strides = */ {1}),
            std::nullopt);
  EXPECT_EQ(getAccessRangeExtent({2, 0, 1}, {16, 4, 1}), std::nullopt);
}

TEST(GetAccessRangeExtentTest, NegativeStride) {
  EXPECT_EQ(getAccessRangeExtent(/* sizes = */ {1}, /* strides = */ {-1}),
            std::nullopt);
  EXPECT_EQ(getAccessRangeExtent({2, 4, 4}, {16, -4, 1}), std::nullopt);
}

TEST(GetAccessRangeExtentTest, ZeroStride) {
  EXPECT_EQ(getAccessRangeExtent({1}, /* strides = */ {0}), std::nullopt);
  EXPECT_EQ(getAccessRangeExtent({2, 4, 4}, {16, 0, 1}), std::nullopt);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
