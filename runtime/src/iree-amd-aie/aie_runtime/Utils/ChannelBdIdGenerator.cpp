// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/aie_runtime/Utils/ChannelBdIdGenerator.h"

namespace mlir::iree_compiler::AMDAIE {

std::optional<uint32_t> ChannelBdIdGenerator::getAndAssignBdId(
    uint32_t channel, BdIdAssignmentMode mode) {
  if (!channelToValidBdIds.contains(channel) ||
      channelToValidBdIds[channel].empty()) {
    return std::nullopt;
  }

  if (mode == BdIdAssignmentMode::Smallest) {
    // Smallest: Find the smallest unassigned BD id
    for (uint32_t bdId : channelToValidBdIds[channel]) {
      if (!isBdIdAssigned(bdId)) {
        assignBdId(bdId);
        return bdId;
      }
    }
  } else if (mode == BdIdAssignmentMode::Incremental) {
    // Incremental: Find the first unassigned BD id greater than lastUsedBdId,
    for (uint32_t bdId : channelToValidBdIds[channel]) {
      if (bdId > lastUsedBdId && !isBdIdAssigned(bdId)) {
        assignBdId(bdId);
        return bdId;
      }
    }
    // If not found, wrap around and check again
    for (uint32_t bdId : channelToValidBdIds[channel]) {
      if (bdId <= lastUsedBdId && !isBdIdAssigned(bdId)) {
        assignBdId(bdId);
        return bdId;
      }
    }
  } else {
    assert(false && "Unsupported BdIdAssignmentMode");
  }

  // No valid BD id found
  return std::nullopt;
}

}  // namespace mlir::iree_compiler::AMDAIE
