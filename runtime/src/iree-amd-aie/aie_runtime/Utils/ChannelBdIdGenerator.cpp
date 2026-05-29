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

SmallVector<uint32_t> ChannelBdIdGenerator::getAndAssignConsecutiveBdIds(
    uint32_t channel, uint32_t num) {
  SmallVector<uint32_t> result;
  if (num == 0 || !channelToValidBdIds.contains(channel)) return result;
  const SmallVector<uint32_t> &valid = channelToValidBdIds[channel];
  // Scan the valid ids in order and track the longest run of consecutive
  // (contiguous-value) unassigned ids, stopping early once a run reaches `num`.
  // Scanning ascending makes `best` the lowest-offset such run.
  SmallVector<uint32_t> best, cur;
  for (uint32_t id : valid) {
    if (isBdIdAssigned(id)) {
      cur.clear();
      continue;
    }
    if (!cur.empty() && id == cur.back() + 1)
      cur.push_back(id);
    else
      cur.assign(1, id);
    if (cur.size() > best.size()) best = cur;
    if (best.size() >= num) break;
  }
  uint32_t n = std::min<uint32_t>(best.size(), num);
  for (uint32_t i = 0; i < n; ++i) {
    assignBdId(best[i]);
    result.push_back(best[i]);
  }
  return result;
}

}  // namespace mlir::iree_compiler::AMDAIE
