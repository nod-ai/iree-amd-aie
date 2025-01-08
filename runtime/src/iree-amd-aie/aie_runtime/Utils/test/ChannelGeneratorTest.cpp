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

TEST(ChannelGeneratorTest, GetFirstAvailable) {
  ChannelGenerator generator(2, 2);
  EXPECT_EQ(
      generator.getProducerDMAChannel(ChannelAssignmentMode::FirstAvailable)
          .value(),
      0);
  EXPECT_EQ(
      generator.getConsumerDMAChannel(ChannelAssignmentMode::FirstAvailable)
          .value(),
      0);
  EXPECT_EQ(
      generator.getProducerDMAChannel(ChannelAssignmentMode::FirstAvailable)
          .value(),
      0);
  EXPECT_EQ(
      generator.getConsumerDMAChannel(ChannelAssignmentMode::FirstAvailable)
          .value(),
      0);
  EXPECT_EQ(
      generator.getProducerDMAChannel(ChannelAssignmentMode::FirstAvailable)
          .value(),
      0);
  EXPECT_EQ(
      generator.getConsumerDMAChannel(ChannelAssignmentMode::FirstAvailable)
          .value(),
      0);
}

TEST(ChannelGeneratorTest, GetRoundRobin) {
  ChannelGenerator generator(2, 2);
  EXPECT_EQ(generator.getProducerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            0);
  EXPECT_EQ(generator.getConsumerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            0);
  EXPECT_EQ(generator.getProducerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            1);
  EXPECT_EQ(generator.getConsumerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            1);
  EXPECT_EQ(generator.getProducerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            0);
  EXPECT_EQ(generator.getConsumerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            0);
  EXPECT_EQ(generator.getProducerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            1);
  EXPECT_EQ(generator.getConsumerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            1);
}

TEST(ChannelGeneratorTest, GetAssign) {
  ChannelGenerator generator(2, 2);
  EXPECT_EQ(generator.getProducerDMAChannel().value(), 0);
  generator.assignProducerDMAChannel(0);
  EXPECT_EQ(generator.getConsumerDMAChannel().value(), 0);
  generator.assignConsumerDMAChannel(0);
  EXPECT_EQ(generator.getProducerDMAChannel().value(), 1);
  generator.assignProducerDMAChannel(1);
  EXPECT_EQ(generator.getConsumerDMAChannel().value(), 1);
  generator.assignConsumerDMAChannel(1);
  EXPECT_EQ(generator.getProducerDMAChannel(), std::nullopt);
  EXPECT_EQ(generator.getConsumerDMAChannel(), std::nullopt);
}

TEST(ChannelGeneratorTest, Occupied) {
  ChannelGenerator generator(4, 4);
  generator.assignProducerDMAChannel(0);
  generator.assignConsumerDMAChannel(0);
  generator.assignProducerDMAChannel(2);
  generator.assignConsumerDMAChannel(2);
  EXPECT_EQ(
      generator.getProducerDMAChannel(ChannelAssignmentMode::FirstAvailable)
          .value(),
      1);
  EXPECT_EQ(
      generator.getConsumerDMAChannel(ChannelAssignmentMode::FirstAvailable)
          .value(),
      1);
  EXPECT_EQ(
      generator.getProducerDMAChannel(ChannelAssignmentMode::FirstAvailable)
          .value(),
      1);
  EXPECT_EQ(
      generator.getConsumerDMAChannel(ChannelAssignmentMode::FirstAvailable)
          .value(),
      1);
  EXPECT_EQ(generator.getProducerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            3);
  EXPECT_EQ(generator.getConsumerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            3);
  EXPECT_EQ(generator.getProducerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            1);
  EXPECT_EQ(generator.getConsumerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            1);
  EXPECT_EQ(generator.getProducerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            3);
  EXPECT_EQ(generator.getConsumerDMAChannel(ChannelAssignmentMode::RoundRobin)
                .value(),
            3);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
