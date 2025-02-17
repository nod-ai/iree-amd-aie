// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_UTILS_CHANNEL_GENERATOR_H_
#define IREE_COMPILER_AMDAIE_UTILS_CHANNEL_GENERATOR_H_

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

using namespace llvm;

namespace mlir::iree_compiler::AMDAIE {

enum class ChannelAssignmentMode {
  FirstAvailableCircuitFlow,
  FirstAvailablePacketFlow,
  RoundRobinPacketFlow
};

/// Utility to generate valid channels.
class ChannelGenerator {
 public:
  ChannelGenerator() {}
  ChannelGenerator(uint8_t numProducerChannels, uint8_t numConsumerChannels)
      : numProducerChannels(numProducerChannels),
        numConsumerChannels(numConsumerChannels) {
    assert(numProducerChannels > 0 && numConsumerChannels > 0 &&
           "Invalid number of producer/consumer channels.");
  }

  /// Attempts to find the first available channel that is not present in any of
  /// the given exclusion sets.
  std::optional<uint8_t> findFirstAvailableChannel(
      uint8_t numChannels,
      ArrayRef<llvm::SmallSetVector<uint8_t, 8>> excludeSets);

  /// Retrieves the next producer channel using the specified strategy.
  std::optional<uint8_t> getAndAssignProducerDMAChannel(
      ChannelAssignmentMode mode);

  /// Retrieves the next consumer channel using the specified strategy.
  std::optional<uint8_t> getAndAssignConsumerDMAChannel(
      ChannelAssignmentMode mode);

  /// Assigns the provided producer channel.
  void assignProducerDMAChannel(uint8_t channel, ChannelAssignmentMode mode);

  /// Assigns the provided consumer channel.
  void assignConsumerDMAChannel(uint8_t channel, ChannelAssignmentMode mode);

 private:
  uint8_t numProducerChannels = 0;
  uint8_t numConsumerChannels = 0;
  // Tracks the channels that are used by circuit flows.
  llvm::SmallSetVector<uint8_t, 8> assignedCircuitProducerChannels;
  llvm::SmallSetVector<uint8_t, 8> assignedCircuitConsumerChannels;
  // Tracks the channels that are used by packet flows.
  llvm::SmallSetVector<uint8_t, 8> assignedPacketProducerChannels;
  llvm::SmallSetVector<uint8_t, 8> assignedPacketConsumerChannels;
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_UTILS_CHANNEL_GENERATOR_H_
