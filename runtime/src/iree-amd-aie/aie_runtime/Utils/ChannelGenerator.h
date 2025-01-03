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

/// Utility to generate valid channels.
class ChannelGenerator {
 public:
  ChannelGenerator() {}
  ChannelGenerator(uint8_t numProducerChannels, uint8_t numConsumerChannels)
      : numProducerChannels(numProducerChannels),
        numConsumerChannels(numConsumerChannels) {}

  /// Returns its next usable producer channel.
  std::optional<uint8_t> getAndAssignProducerDMAChannel() {
    for (uint8_t i = 0; i < numProducerChannels; i++) {
      if (!assignedProducerChannels.count(i)) {
        assignedProducerChannels.insert(i);
        return i;
      }
    }
    return std::nullopt;
  }

  /// Returns its next usable consumer channel.
  std::optional<uint8_t> getAndAssignConsumerDMAChannel() {
    for (uint8_t i = 0; i < numConsumerChannels; i++) {
      if (!assignedConsumerChannels.count(i)) {
        assignedConsumerChannels.insert(i);
        return i;
      }
    }
    return std::nullopt;
  }

  /// Assigns the provided producer channel.
  void assignProducerDMAChannel(uint8_t channel) {
    assignedProducerChannels.insert(channel);
  }

  /// Assigns the provided consumer channel.
  void assignConsumerDMAChannel(uint8_t channel) {
    assignedConsumerChannels.insert(channel);
  }

 private:
  uint8_t numProducerChannels = 0;
  uint8_t numConsumerChannels = 0;
  DenseSet<uint8_t> assignedProducerChannels;
  DenseSet<uint8_t> assignedConsumerChannels;
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_UTILS_CHANNEL_GENERATOR_H_
