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

DenseMap<uint32_t, SmallVector<uint32_t>> getTestRangeChannelToValidBdIds(
    uint32_t n) {
  SmallVector<uint32_t> range(n);
  std::iota(range.begin(), range.end(), 0);
  return {{0, range}, {1, range}};
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

TEST(ChannelBdIdGeneratorTest, IncrementalAssignAll) {
  ChannelBdIdGenerator generator(getTestSingleRangeChannelToValidBdIds());
  EXPECT_EQ(
      generator.getAndAssignBdId(0, BdIdAssignmentMode::Incremental).value(),
      0);
  EXPECT_EQ(
      generator.getAndAssignBdId(0, BdIdAssignmentMode::Incremental).value(),
      1);
  EXPECT_EQ(
      generator.getAndAssignBdId(0, BdIdAssignmentMode::Incremental).value(),
      2);
  EXPECT_EQ(generator.getAndAssignBdId(0, BdIdAssignmentMode::Incremental),
            std::nullopt);
}

TEST(ChannelBdIdGeneratorTest, IncrementalWrap) {
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
  EXPECT_EQ(
      generator.getAndAssignBdId(0, BdIdAssignmentMode::Incremental).value(),
      0);
  generator.releaseBdId(0);
}

TEST(ChannelBdIdGeneratorTest, ConsecutiveBlockFull) {
  ChannelBdIdGenerator generator(getTestRangeChannelToValidBdIds(8));
  EXPECT_EQ(generator.getAndAssignConsecutiveBdIds(0, 3),
            SmallVector<uint32_t>({0, 1, 2}));
  EXPECT_TRUE(generator.isBdIdAssigned(0));
  EXPECT_TRUE(generator.isBdIdAssigned(2));
  // The next block continues after the first.
  EXPECT_EQ(generator.getAndAssignConsecutiveBdIds(0, 3),
            SmallVector<uint32_t>({3, 4, 5}));
}

TEST(ChannelBdIdGeneratorTest, ConsecutiveBlockShorterWhenFragmented) {
  ChannelBdIdGenerator generator(getTestRangeChannelToValidBdIds(4));
  // Reserve id 2, splitting the pool into {0, 1} and {3}.
  generator.assignBdId(2);
  // A run of 4 doesn't exist; the longest available run is {0, 1}.
  EXPECT_EQ(generator.getAndAssignConsecutiveBdIds(0, 4),
            SmallVector<uint32_t>({0, 1}));
}

TEST(ChannelBdIdGeneratorTest, ConsecutiveBlockNeverWrapsOutOfRange) {
  // Regression: a plain incremental allocation marches `lastUsedBdId` toward
  // the top of the pool and then wraps, yielding a non-consecutive set (e.g.
  // [3, 0]) whose use as `offset + iv % n` would emit out-of-range ids. The
  // consecutive allocator must always return an in-range, consecutive run.
  ChannelBdIdGenerator generator(getTestRangeChannelToValidBdIds(4));  // 0..3
  generator.getAndAssignBdId(0, BdIdAssignmentMode::Incremental);      // 0
  generator.getAndAssignBdId(0, BdIdAssignmentMode::Incremental);      // 1
  generator.getAndAssignBdId(0, BdIdAssignmentMode::Incremental);      // 2
  generator.releaseBdId(0);
  generator.releaseBdId(1);
  // Free now: {0, 1, 3}, incremental cursor at 2 (an incremental block of 2
  // would wrap to [3, 0]). The consecutive run must be the in-range [0, 1].
  SmallVector<uint32_t> block = generator.getAndAssignConsecutiveBdIds(0, 2);
  EXPECT_EQ(block, SmallVector<uint32_t>({0, 1}));
  for (uint32_t id : block) EXPECT_LT(id, 4u);
}

TEST(ChannelBdIdGeneratorTest, ConsecutiveBlockSingleWhenFullyFragmented) {
  ChannelBdIdGenerator generator(getTestRangeChannelToValidBdIds(4));  // 0..3
  // Reserve 1 and 2, leaving only the isolated singles {0} and {3}.
  generator.assignBdId(1);
  generator.assignBdId(2);
  // No run of length > 1 exists; the lowest single is returned (this is what
  // collapses `AMDAIEAssignNpuDmaBdIds` to its constant-id branch).
  EXPECT_EQ(generator.getAndAssignConsecutiveBdIds(0, 3),
            SmallVector<uint32_t>({0}));
}

TEST(ChannelBdIdGeneratorTest, ConsecutiveBlockEmptyWhenExhausted) {
  ChannelBdIdGenerator generator(getTestRangeChannelToValidBdIds(2));  // 0..1
  generator.assignBdId(0);
  generator.assignBdId(1);
  // Channel fully assigned -> empty (drives the pass's failure path).
  EXPECT_TRUE(generator.getAndAssignConsecutiveBdIds(0, 2).empty());
  // Unknown channel -> empty too.
  EXPECT_TRUE(generator.getAndAssignConsecutiveBdIds(7, 1).empty());
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
