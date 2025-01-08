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

TEST(ChannelGeneratorTest, GetAssign) {
  ChannelGenerator generator(2, 2);
  bool isPacketFlow = false;
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(isPacketFlow).value(), 0);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(isPacketFlow).value(), 0);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(isPacketFlow).value(), 1);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(isPacketFlow).value(), 1);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(isPacketFlow),
            std::nullopt);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(isPacketFlow),
            std::nullopt);
}

TEST(ChannelGeneratorTest, Occupied) {
  ChannelGenerator generator(4, 4);
  bool isPacketFlow = false;
  generator.assignProducerDMAChannel(0);
  generator.assignConsumerDMAChannel(0);
  generator.assignProducerDMAChannel(2);
  generator.assignConsumerDMAChannel(2);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(isPacketFlow).value(), 1);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(isPacketFlow).value(), 1);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(isPacketFlow).value(), 3);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(isPacketFlow).value(), 3);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(isPacketFlow),
            std::nullopt);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(isPacketFlow),
            std::nullopt);
}

TEST(ChannelGeneratorTest, PacketFlow) {
  ChannelGenerator generator(4, 4);
  generator.assignProducerDMAChannel(0);
  generator.assignConsumerDMAChannel(0);
  generator.assignProducerDMAChannel(2);
  generator.assignConsumerDMAChannel(2);
  bool isPacketFlow = true;
  // Packet flow should not occupy the channel exclusively, and the available
  // channel appears in a round-robin fashion.
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(isPacketFlow).value(), 1);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(isPacketFlow).value(), 1);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(isPacketFlow).value(), 3);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(isPacketFlow).value(), 3);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(isPacketFlow).value(), 1);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(isPacketFlow).value(), 1);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(isPacketFlow).value(), 3);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(isPacketFlow).value(), 3);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
