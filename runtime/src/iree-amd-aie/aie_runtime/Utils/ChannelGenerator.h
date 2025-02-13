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
      ArrayRef<llvm::SmallSetVector<uint8_t, 8>> excludeSets) {
    for (uint8_t channel = 0; channel < numChannels; ++channel) {
      if (llvm::none_of(
              excludeSets,
              [&](const llvm::SmallSetVector<uint8_t, 8> &excludeSet) {
                return excludeSet.count(channel);
              })) {
        return channel;
      }
    }
    return std::nullopt;
  }

  /// Retrieves the next producer channel using the specified strategy.
  std::optional<uint8_t> getAndAssignProducerDMAChannel(
      ChannelAssignmentMode mode) {
    std::optional<uint8_t> channel;
    switch (mode) {
      // Select the first available channel for circuit flow.
      // A channel is valid if it is not already assigned to any circuit or
      // packet flow.
      case ChannelAssignmentMode::FirstAvailableCircuitFlow: {
        channel = findFirstAvailableChannel(
            numProducerChannels,
            {assignedCircuitProducerChannels, assignedPacketProducerChannels});
        break;
      }
      // Select the first available channel for packet flow.
      // A channel is valid if it is not already assigned to a circuit flow.
      case ChannelAssignmentMode::FirstAvailablePacketFlow: {
        channel = findFirstAvailableChannel(numProducerChannels,
                                            {assignedCircuitProducerChannels});
        break;
      }
      // Select the channel for packet flow, using a round-robin strategy for
      // load balancing:
      // 1. Prefer an unused channel (not assigned to any circuit or packet
      // flow).
      // 2. If no such channel is available, reuse the least recently used
      // packet flow channel from `assignedPacketProducerChannels.front()`.
      case ChannelAssignmentMode::RoundRobinPacketFlow: {
        channel = findFirstAvailableChannel(
            numProducerChannels,
            {assignedCircuitProducerChannels, assignedPacketProducerChannels});
        if (!channel && !assignedPacketProducerChannels.empty())
          channel = assignedPacketProducerChannels.front();
        break;
      }
      default:
        assert(false && "Unsupported ChannelAssignmentMode");
    }
    // Assign the channel if found.
    if (channel.has_value()) assignProducerDMAChannel(channel.value(), mode);
    return channel;
  }

  /// Retrieves the next consumer channel using the specified strategy.
  std::optional<uint8_t> getAndAssignConsumerDMAChannel(
      ChannelAssignmentMode mode) {
    std::optional<uint8_t> channel;
    switch (mode) {
      // Select the first available channel for circuit flow.
      // A channel is valid if it is not already assigned to any circuit or
      // packet flow.
      case ChannelAssignmentMode::FirstAvailableCircuitFlow: {
        channel = findFirstAvailableChannel(
            numConsumerChannels,
            {assignedCircuitConsumerChannels, assignedPacketConsumerChannels});
        break;
      }
      // Select the first available channel for packet flow.
      // A channel is valid if it is not already assigned to a circuit flow.
      case ChannelAssignmentMode::FirstAvailablePacketFlow: {
        channel = findFirstAvailableChannel(numConsumerChannels,
                                            {assignedCircuitConsumerChannels});
        break;
      }
      // Select the channel for packet flow, using a round-robin strategy for
      // load balancing:
      // 1. Prefer an unused channel (not assigned to any circuit or packet
      // flow).
      // 2. If no such channel is available, reuse the least recently used
      // packet flow channel from `assignedPacketConsumerChannels.front()`.
      case ChannelAssignmentMode::RoundRobinPacketFlow: {
        channel = findFirstAvailableChannel(
            numConsumerChannels,
            {assignedCircuitConsumerChannels, assignedPacketConsumerChannels});
        if (!channel && !assignedPacketConsumerChannels.empty())
          channel = assignedPacketConsumerChannels.front();
        break;
      }
      default:
        assert(false && "Unsupported ChannelAssignmentMode");
    }
    // Assign the channel if found.
    if (channel.has_value()) assignConsumerDMAChannel(channel.value(), mode);
    return channel;
  }

  /// Assigns the provided producer channel.
  void assignProducerDMAChannel(uint8_t channel, ChannelAssignmentMode mode) {
    switch (mode) {
      case ChannelAssignmentMode::FirstAvailableCircuitFlow:
        assignedCircuitProducerChannels.insert(channel);
        break;
      case ChannelAssignmentMode::FirstAvailablePacketFlow:
        assignedPacketProducerChannels.insert(channel);
        break;
      case ChannelAssignmentMode::RoundRobinPacketFlow:
        // Remove and reinsert to update the least recently used channel
        // (front).
        assignedPacketProducerChannels.remove(channel);
        assignedPacketProducerChannels.insert(channel);
        break;
      default:
        assert(false && "Unsupported ChannelAssignmentMode");
    }
  }

  /// Assigns the provided consumer channel.
  void assignConsumerDMAChannel(uint8_t channel, ChannelAssignmentMode mode) {
    switch (mode) {
      case ChannelAssignmentMode::FirstAvailableCircuitFlow:
        assignedCircuitConsumerChannels.insert(channel);
        break;
      case ChannelAssignmentMode::FirstAvailablePacketFlow:
        assignedPacketConsumerChannels.insert(channel);
        break;
      case ChannelAssignmentMode::RoundRobinPacketFlow:
        // Remove and reinsert to update the least recently used channel
        // (front).
        assignedPacketConsumerChannels.remove(channel);
        assignedPacketConsumerChannels.insert(channel);
        break;
      default:
        assert(false && "Unsupported ChannelAssignmentMode");
    }
  }

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
