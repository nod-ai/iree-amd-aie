// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "gtest/gtest.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelBdIdGenerator.h"

namespace {

using namespace mlir::iree_compiler::AMDAIE;

DenseMap<uint32_t, SmallVector<uint32_t>>
getTestSingleRangeChannelToValidBdIds() {
  SmallVector<uint32_t> range(3);
  std::iota(range.begin(), range.end(), 0);
  DenseMap<uint32_t, SmallVector<uint32_t>> channelToValidBdIds = {{0, range},
                                                                   {1, range}};
  return channelToValidBdIds;
}

DenseMap<uint32_t, SmallVector<uint32_t>> getTestEvenOddChannelToValidBdIds() {
  SmallVector<uint32_t> evenRange(4);
  std::iota(evenRange.begin(), evenRange.end(), 0);
  SmallVector<uint32_t> oddRange(4);
  std::iota(oddRange.begin(), oddRange.end(), 4);
  DenseMap<uint32_t, SmallVector<uint32_t>> channelToValidBdIds = {
      {0, evenRange}, {1, oddRange},  {2, evenRange},
      {3, oddRange},  {4, evenRange}, {5, oddRange}};
  return channelToValidBdIds;
}

TEST(ChannelBdIdGeneratorTest, SingleRange) {
  ChannelBdIdGenerator generator(getTestSingleRangeChannelToValidBdIds());
  EXPECT_EQ(generator.getAndAssignBdId(0).value(), 0);
  EXPECT_EQ(generator.isBdIdAssigned(0), true);
  EXPECT_EQ(generator.getAndAssignBdId(1).value(), 1);
  EXPECT_EQ(generator.isBdIdAssigned(1), true);
  EXPECT_EQ(generator.getAndAssignBdId(0).value(), 2);
  EXPECT_EQ(generator.isBdIdAssigned(2), true);
  EXPECT_EQ(generator.getAndAssignBdId(1), std::nullopt);
}

TEST(ChannelBdIdGeneratorTest, EvenOdd) {
  ChannelBdIdGenerator generator(getTestEvenOddChannelToValidBdIds());
  // Check that even channel BDs start from 0
  EXPECT_EQ(generator.getAndAssignBdId(0).value(), 0);
  EXPECT_EQ(generator.isBdIdAssigned(0), true);
  // Check that odd channel BDs start from 4
  EXPECT_EQ(generator.getAndAssignBdId(1).value(), 4);
  EXPECT_EQ(generator.isBdIdAssigned(4), true);
  // Check assignment of other even BDs
  EXPECT_EQ(generator.getAndAssignBdId(2).value(), 1);
  EXPECT_EQ(generator.isBdIdAssigned(1), true);
  EXPECT_EQ(generator.getAndAssignBdId(4).value(), 2);
  EXPECT_EQ(generator.isBdIdAssigned(2), true);
  EXPECT_EQ(generator.getAndAssignBdId(0).value(), 3);
  EXPECT_EQ(generator.isBdIdAssigned(3), true);
  EXPECT_EQ(generator.getAndAssignBdId(2), std::nullopt);
  // Check assignment of other odd BDs
  EXPECT_EQ(generator.getAndAssignBdId(3).value(), 5);
  EXPECT_EQ(generator.isBdIdAssigned(5), true);
  EXPECT_EQ(generator.getAndAssignBdId(5).value(), 6);
  EXPECT_EQ(generator.isBdIdAssigned(6), true);
  EXPECT_EQ(generator.getAndAssignBdId(1).value(), 7);
  EXPECT_EQ(generator.isBdIdAssigned(7), true);
  EXPECT_EQ(generator.getAndAssignBdId(3), std::nullopt);
}

TEST(ChannelBdIdGeneratorTest, AssignBdId) {
  ChannelBdIdGenerator generator(getTestSingleRangeChannelToValidBdIds());
  generator.assignBdId(0);
  EXPECT_EQ(generator.getAndAssignBdId(0).value(), 1);
  EXPECT_EQ(generator.isBdIdAssigned(1), true);
  generator.assignBdId(2);
  EXPECT_EQ(generator.getAndAssignBdId(1), std::nullopt);
}

TEST(ChannelBdIdGeneratorTest, Release) {
  ChannelBdIdGenerator generator(getTestSingleRangeChannelToValidBdIds());
  EXPECT_EQ(generator.getAndAssignBdId(0).value(), 0);
  EXPECT_EQ(generator.isBdIdAssigned(0), true);
  generator.releaseBdId(0);
  EXPECT_EQ(generator.getAndAssignBdId(1).value(), 0);
  EXPECT_EQ(generator.isBdIdAssigned(0), true);
  EXPECT_EQ(generator.getAndAssignBdId(0).value(), 1);
  EXPECT_EQ(generator.isBdIdAssigned(1), true);
  generator.releaseBdId(1);
  EXPECT_EQ(generator.getAndAssignBdId(1).value(), 1);
  EXPECT_EQ(generator.isBdIdAssigned(1), true);
}

TEST(ChannelBdIdGeneratorTest, IncrementalAssign) {
  ChannelBdIdGenerator generator(getTestSingleRangeChannelToValidBdIds());
  EXPECT_EQ(
      generator.getAndAssignBdId(0, BdIdAssignmentMode::Incremental).value(),
      0);
  generator.releaseBdId(0);
  EXPECT_EQ(
      generator.getAndAssignBdId(0, BdIdAssignmentMode::Incremental).value(),
      1);
  generator.releaseBdId(1);
  EXPECT_EQ(
      generator.getAndAssignBdId(0, BdIdAssignmentMode::Incremental).value(),
      2);
  generator.releaseBdId(2);
  EXPECT_EQ(generator.getAndAssignBdId(0).value(), 0);
  generator.releaseBdId(0);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
