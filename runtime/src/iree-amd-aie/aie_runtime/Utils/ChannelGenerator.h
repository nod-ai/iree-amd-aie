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
/// TODO(jornt): add physical feasibility checks on channels.
class ChannelGenerator {
 public:
  ChannelGenerator() {}

  /// Given a tile, returns its next usable producer channel.
  uint8_t getProducerDMAChannel(Value tile) {
    return producerChannelsPerTile[tile]++;
  }

  /// Given a tile, returns its next usable consumer channel.
  uint8_t getConsumerDMAChannel(Value tile) {
    return consumerChannelsPerTile[tile]++;
  }

 private:
  DenseMap<Value, uint8_t> producerChannelsPerTile;
  DenseMap<Value, uint8_t> consumerChannelsPerTile;
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_UTILS_CHANNEL_GENERATOR_H_
