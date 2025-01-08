// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_UTILS_CHANNEL_GENERATOR_H_
#define IREE_COMPILER_AMDAIE_UTILS_CHANNEL_GENERATOR_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

using namespace llvm;

namespace mlir::iree_compiler::AMDAIE {

enum class ChannelAssignmentMode { FirstAvailable, RoundRobin };

/// Utility to generate valid channels.
class ChannelGenerator {
 public:
  ChannelGenerator() {}
  ChannelGenerator(uint8_t numProducerChannels, uint8_t numConsumerChannels)
      : numProducerChannels(numProducerChannels),
        numConsumerChannels(numConsumerChannels) {
    assert(numProducerChannels > 0 && numConsumerChannels > 0 &&
           "Invalid number of producer/consumer channels.");
    // Initialize to the last channel for round-robin usage.
    lastUsedProducerChannel = numProducerChannels - 1;
    lastUsedConsumerChannel = numConsumerChannels - 1;
  }

  /// Returns its next usable producer channel. By default, it uses round-robin
  /// for load balancing.
  std::optional<uint8_t> getProducerDMAChannel(
      ChannelAssignmentMode mode = ChannelAssignmentMode::RoundRobin) {
    for (uint8_t offset = 1; offset <= numProducerChannels; ++offset) {
      uint8_t i;
      if (mode == ChannelAssignmentMode::FirstAvailable) {
        i = offset - 1;
      } else if (mode == ChannelAssignmentMode::RoundRobin) {
        i = (lastUsedProducerChannel + offset) % numProducerChannels;
      } else {
        assert(false && "Unsupported ChannelAssignmentMode");
      }
      if (!assignedProducerChannels.count(i)) {
        lastUsedProducerChannel = i;
        return i;
      }
    }
    return std::nullopt;
  }

  /// Returns its next usable consumer channel. By default, it uses round-robin
  /// for load balancing.
  std::optional<uint8_t> getConsumerDMAChannel(
      ChannelAssignmentMode mode = ChannelAssignmentMode::RoundRobin) {
    for (uint8_t offset = 1; offset <= numConsumerChannels; ++offset) {
      uint8_t i;
      if (mode == ChannelAssignmentMode::FirstAvailable) {
        i = offset - 1;
      } else if (mode == ChannelAssignmentMode::RoundRobin) {
        i = (lastUsedConsumerChannel + offset) % numConsumerChannels;
      } else {
        assert(false && "Unsupported ChannelAssignmentMode");
      }
      if (!assignedConsumerChannels.count(i)) {
        lastUsedConsumerChannel = i;
        return i;
      }
    }
    return std::nullopt;
  }

  /// Assigns the provided producer channel, only used for circuit flow.
  void assignProducerDMAChannel(uint8_t channel) {
    assignedProducerChannels.insert(channel);
  }

  /// Assigns the provided consumer channel, only used for circuit flow.
  void assignConsumerDMAChannel(uint8_t channel) {
    assignedConsumerChannels.insert(channel);
  }

 private:
  uint8_t numProducerChannels = 0;
  uint8_t numConsumerChannels = 0;
  // Tracks the channels that are used by circuit flows.
  DenseSet<uint8_t> assignedProducerChannels;
  DenseSet<uint8_t> assignedConsumerChannels;
  // Tracks the last used channel, for both circuit and packet flows.
  uint8_t lastUsedProducerChannel = 0;
  uint8_t lastUsedConsumerChannel = 0;
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_UTILS_CHANNEL_GENERATOR_H_
