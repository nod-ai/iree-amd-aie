// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/aie_runtime/Utils/ChannelBdIdGenerator.h"

namespace mlir::iree_compiler::AMDAIE {

std::optional<uint32_t> ChannelBdIdGenerator::getAndAssignBdId(
    uint32_t channel) {
  if (!channelToValidBdIds.contains(channel) ||
      channelToValidBdIds[channel].empty()) {
    return std::nullopt;
  }
  uint32_t bdId = channelToValidBdIds[channel][0];
  size_t index{1};
  while (isBdIdAssigned(bdId) && index < channelToValidBdIds[channel].size()) {
    bdId = channelToValidBdIds[channel][index++];
  }
  if (isBdIdAssigned(bdId)) return std::nullopt;
  assignBdId(bdId);
  return bdId;
}

}  // namespace mlir::iree_compiler::AMDAIE
