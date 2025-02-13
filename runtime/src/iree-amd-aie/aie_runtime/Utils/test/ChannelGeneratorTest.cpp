// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "gtest/gtest.h"
#include "iree-amd-aie/aie_runtime/Utils/ChannelGenerator.h"

namespace {

using namespace mlir::iree_compiler::AMDAIE;

TEST(ChannelGeneratorTest, GetAssignFirstAvailableCircuitFlow) {
  ChannelGenerator generator(2, 2);
  // Keep incrementing the channel number until all channels are assigned.
  ChannelAssignmentMode mode = ChannelAssignmentMode::FirstAvailableCircuitFlow;
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 0);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 0);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 1);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 1);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode), std::nullopt);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode), std::nullopt);
}

TEST(ChannelGeneratorTest, GetAssignFirstAvailablePacketFlow) {
  ChannelGenerator generator(2, 2);
  // Use the same channel number, as it can be assigned to multiple packet
  // flows.
  ChannelAssignmentMode mode = ChannelAssignmentMode::FirstAvailablePacketFlow;
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 0);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 0);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 0);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 0);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 0);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 0);
}

TEST(ChannelGeneratorTest, GetAssignRoundRobinPacketFlow) {
  ChannelGenerator generator(2, 2);
  // Round-robin between the availble two channels, for load balancing.
  ChannelAssignmentMode mode = ChannelAssignmentMode::RoundRobinPacketFlow;
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 0);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 0);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 1);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 1);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 0);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 0);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 1);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 1);
}

TEST(ChannelGeneratorTest, Occupied) {
  ChannelGenerator generator(4, 4);
  // Reserve channels 0 for circuit flow.
  ChannelAssignmentMode mode = ChannelAssignmentMode::FirstAvailableCircuitFlow;
  generator.assignProducerDMAChannel(0, mode);
  generator.assignConsumerDMAChannel(0, mode);
  // Reserve channels 1 for packet flow.
  mode = ChannelAssignmentMode::FirstAvailablePacketFlow;
  generator.assignProducerDMAChannel(1, mode);
  generator.assignConsumerDMAChannel(1, mode);
  // The next available channel for circuit flow is 2.
  mode = ChannelAssignmentMode::FirstAvailableCircuitFlow;
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 2);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 2);
  // Channel 0 and 2 are already assigned for circuit flow. Therefore, for
  // packet flow, the next available channel is round-robin between 1 and 3.
  mode = ChannelAssignmentMode::RoundRobinPacketFlow;
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 3);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 3);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 1);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 1);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(mode).value(), 3);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(mode).value(), 3);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
