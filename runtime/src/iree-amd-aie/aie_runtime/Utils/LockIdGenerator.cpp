// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/aie_runtime/Utils/LockIdGenerator.h"

namespace mlir::iree_compiler::AMDAIE {

std::optional<uint8_t> LockIdGenerator::getAndAssignLockId(uint32_t col,
                                                           uint32_t row) {
  uint32_t maxNumLocks = deviceModel.getNumLocks(col, row);
  uint32_t lockId = 0;
  while (lockId < maxNumLocks && isLockIdAssigned(col, row, lockId)) lockId++;
  if (lockId >= maxNumLocks) return std::nullopt;
  assignLockId(col, row, lockId);
  return lockId;
}

}  // namespace mlir::iree_compiler::AMDAIE
