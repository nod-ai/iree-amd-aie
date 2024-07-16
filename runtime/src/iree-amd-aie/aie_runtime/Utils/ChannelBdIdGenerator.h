// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_UTILS_CHANNEL_BD_ID_GENERATOR_H_
#define IREE_COMPILER_AMDAIE_UTILS_CHANNEL_BD_ID_GENERATOR_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

using namespace llvm;

namespace mlir::iree_compiler::AMDAIE {

/// Utility to generate valid buffer descriptor (BD) ids for channels. Keeps
/// state on assigned BD ids to avoid reuse.
class ChannelBdIdGenerator {
 public:
  ChannelBdIdGenerator() {}
  ChannelBdIdGenerator(
      const DenseMap<uint32_t, SmallVector<uint32_t>> &channelToValidBdIds)
      : channelToValidBdIds(channelToValidBdIds) {}
  ChannelBdIdGenerator(
      DenseMap<uint32_t, SmallVector<uint32_t>> &&channelToValidBdIds)
      : channelToValidBdIds(std::move(channelToValidBdIds)) {}

  void assignBdId(uint32_t bdId) { assignedBdIds.insert(bdId); }

  /// Attempts to find and assign an unused BD id for the provided channel.
  /// Returns `std::nullopt` if no valid BD id could be found.
  std::optional<uint32_t> getAndAssignBdId(uint32_t channel);

  /// Check whether the provided BD id is currently assigned.
  bool isBdIdAssigned(uint32_t bdId) const { return assignedBdIds.count(bdId); }

  /// Releases the provided BD id if it is currently assigned so it can be
  /// reused.
  void releaseBdId(uint32_t bdId) { assignedBdIds.erase(bdId); }

 private:
  // Maps channel indices to vectors of valid BD ids.
  DenseMap<uint32_t, SmallVector<uint32_t>> channelToValidBdIds;
  // Set with all BD ids that are currently assigned.
  DenseSet<uint32_t> assignedBdIds;
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_UTILS_CHANNEL_BD_ID_GENERATOR_H_
