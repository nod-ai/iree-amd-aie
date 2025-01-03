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
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel().value(), 0);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel().value(), 0);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel().value(), 1);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel().value(), 1);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(), std::nullopt);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(), std::nullopt);
}

TEST(ChannelGeneratorTest, Occupied) {
  ChannelGenerator generator(4, 4);
  generator.assignProducerDMAChannel(0);
  generator.assignConsumerDMAChannel(0);
  generator.assignProducerDMAChannel(2);
  generator.assignConsumerDMAChannel(2);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel().value(), 1);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel().value(), 1);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel().value(), 3);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel().value(), 3);
  EXPECT_EQ(generator.getAndAssignProducerDMAChannel(), std::nullopt);
  EXPECT_EQ(generator.getAndAssignConsumerDMAChannel(), std::nullopt);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
