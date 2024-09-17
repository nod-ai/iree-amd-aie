// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_UTILS_LOCK_ID_GENERATOR_H_
#define IREE_COMPILER_AMDAIE_UTILS_LOCK_ID_GENERATOR_H_

#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::iree_compiler::AMDAIE {

/// Utility to generate valid lock ids for usage on tiles.
class LockIdGenerator {
 public:
  LockIdGenerator(const AMDAIEDeviceModel &deviceModel)
      : deviceModel(deviceModel) {}

  void assignLockId(uint32_t col, uint32_t row, uint8_t lockId) {
    tileToAssignedLockIds[TileLoc(col, row)].insert(lockId);
  }

  /// Attempts to find and assign an unused lock id for the provided tile.
  /// Returns `std::nullopt` if no valid lock id could be found.
  std::optional<uint8_t> getAndAssignLockId(uint32_t col, uint32_t row);

  /// Check whether the provided lock id is currently assigned.
  bool isLockIdAssigned(uint32_t col, uint32_t row, uint8_t lockId) const {
    return tileToAssignedLockIds.contains(TileLoc(col, row)) &&
           tileToAssignedLockIds.at(TileLoc(col, row)).contains(lockId);
  }

  /// Releases the provided lock id if it is currently assigned so it can be
  /// reused.
  void releaseLockId(uint32_t col, uint32_t row, uint8_t lockId) {
    tileToAssignedLockIds[TileLoc(col, row)].erase(lockId);
  }

 private:
  // The device model which can be used to look up device related information to
  // ensure correct locks are assigned.
  const AMDAIEDeviceModel &deviceModel;
  // Map to keep track of the assigned lock IDs per tile.
  DenseMap<TileLoc, DenseSet<uint8_t>> tileToAssignedLockIds;
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_UTILS_LOCK_ID_GENERATOR_H_
