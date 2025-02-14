// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/aie_runtime/Utils/ChannelGenerator.h"

namespace mlir::iree_compiler::AMDAIE {

std::optional<uint8_t> ChannelGenerator::findFirstAvailableChannel(
    uint8_t numChannels,
    ArrayRef<llvm::SmallSetVector<uint8_t, 8>> excludeSets) {
  for (uint8_t channel = 0; channel < numChannels; ++channel) {
    if (llvm::none_of(excludeSets,
                      [&](const llvm::SmallSetVector<uint8_t, 8> &excludeSet) {
                        return excludeSet.count(channel);
                      })) {
      return channel;
    }
  }
  return std::nullopt;
}

std::optional<uint8_t> ChannelGenerator::getAndAssignProducerDMAChannel(
    ChannelAssignmentMode mode) {
  std::optional<uint8_t> channel;
  switch (mode) {
    case ChannelAssignmentMode::FirstAvailableCircuitFlow: {
      // Select the first available channel for circuit flow.
      // A channel is valid if it is not already assigned to any circuit or
      // packet flow.
      channel = findFirstAvailableChannel(
          numProducerChannels,
          {assignedCircuitProducerChannels, assignedPacketProducerChannels});
      break;
    }
    case ChannelAssignmentMode::FirstAvailablePacketFlow: {
      // Select the first available channel for packet flow.
      // A channel is valid if it is not already assigned to a circuit flow.
      channel = findFirstAvailableChannel(numProducerChannels,
                                          {assignedCircuitProducerChannels});
      break;
    }
    case ChannelAssignmentMode::RoundRobinPacketFlow: {
      // Select the channel for packet flow, using a round-robin strategy for
      // load balancing:
      // 1. Prefer an unused channel (not assigned to any circuit or packet
      // flow).
      // 2. If no such channel is available, reuse the least recently used
      // packet flow channel from `assignedPacketProducerChannels.front()`.
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

std::optional<uint8_t> ChannelGenerator::getAndAssignConsumerDMAChannel(
    ChannelAssignmentMode mode) {
  std::optional<uint8_t> channel;
  switch (mode) {
    case ChannelAssignmentMode::FirstAvailableCircuitFlow: {
      // Select the first available channel for circuit flow.
      // A channel is valid if it is not already assigned to any circuit or
      // packet flow.
      channel = findFirstAvailableChannel(
          numConsumerChannels,
          {assignedCircuitConsumerChannels, assignedPacketConsumerChannels});
      break;
    }
    case ChannelAssignmentMode::FirstAvailablePacketFlow: {
      // Select the first available channel for packet flow.
      // A channel is valid if it is not already assigned to a circuit flow.
      channel = findFirstAvailableChannel(numConsumerChannels,
                                          {assignedCircuitConsumerChannels});
      break;
    }
    case ChannelAssignmentMode::RoundRobinPacketFlow: {
      // Select the channel for packet flow, using a round-robin strategy for
      // load balancing:
      // 1. Prefer an unused channel (not assigned to any circuit or packet
      // flow).
      // 2. If no such channel is available, reuse the least recently used
      // packet flow channel from `assignedPacketConsumerChannels.front()`.
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

void ChannelGenerator::assignProducerDMAChannel(uint8_t channel,
                                                ChannelAssignmentMode mode) {
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

void ChannelGenerator::assignConsumerDMAChannel(uint8_t channel,
                                                ChannelAssignmentMode mode) {
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

}  // namespace mlir::iree_compiler::AMDAIE
