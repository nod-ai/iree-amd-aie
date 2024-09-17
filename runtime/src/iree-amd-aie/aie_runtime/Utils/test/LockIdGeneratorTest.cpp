// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "gtest/gtest.h"
#include "iree-amd-aie/aie_runtime/Utils/LockIdGenerator.h"

namespace {

using namespace mlir::iree_compiler::AMDAIE;

// Shim has 16 locks in AIE2.
TEST(LockIdGeneratorTest, Shim) {
  AMDAIEDeviceModel deviceModel = getDeviceModel(AMDAIEDevice::npu1_4col);
  LockIdGenerator generator(deviceModel);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(generator.isLockIdAssigned(0, 0, i), false);
    EXPECT_EQ(generator.getAndAssignLockId(0, 0).value(), i);
    EXPECT_EQ(generator.isLockIdAssigned(0, 0, i), true);
  }
  generator.releaseLockId(0, 0, 5);
  EXPECT_EQ(generator.isLockIdAssigned(0, 0, 5), false);
  EXPECT_EQ(generator.getAndAssignLockId(0, 0).value(), 5);
  EXPECT_EQ(generator.isLockIdAssigned(0, 0, 5), true);
  // Out of locks
  EXPECT_EQ(generator.getAndAssignLockId(0, 0), std::nullopt);
  EXPECT_EQ(generator.isLockIdAssigned(0, 0, 16), false);
}

// MemTile has 64 locks in AIE2.
TEST(LockIdGeneratorTest, MemTile) {
  AMDAIEDeviceModel deviceModel = getDeviceModel(AMDAIEDevice::npu1_4col);
  LockIdGenerator generator(deviceModel);
  for (int i = 0; i < 64; i++) {
    EXPECT_EQ(generator.isLockIdAssigned(0, 1, i), false);
    EXPECT_EQ(generator.getAndAssignLockId(0, 1).value(), i);
    EXPECT_EQ(generator.isLockIdAssigned(0, 1, i), true);
  }
  generator.releaseLockId(0, 1, 63);
  EXPECT_EQ(generator.isLockIdAssigned(0, 1, 63), false);
  EXPECT_EQ(generator.getAndAssignLockId(0, 1).value(), 63);
  EXPECT_EQ(generator.isLockIdAssigned(0, 1, 63), true);
  // Out of locks
  EXPECT_EQ(generator.getAndAssignLockId(0, 1), std::nullopt);
  EXPECT_EQ(generator.isLockIdAssigned(0, 1, 64), false);
}

// Core has 16 locks in AIE2.
TEST(LockIdGeneratorTest, Core) {
  AMDAIEDeviceModel deviceModel = getDeviceModel(AMDAIEDevice::npu1_4col);
  LockIdGenerator generator(deviceModel);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(generator.isLockIdAssigned(0, 2, i), false);
    EXPECT_EQ(generator.getAndAssignLockId(0, 2).value(), i);
    EXPECT_EQ(generator.isLockIdAssigned(0, 2, i), true);
  }
  generator.releaseLockId(0, 2, 0);
  EXPECT_EQ(generator.isLockIdAssigned(0, 2, 0), false);
  EXPECT_EQ(generator.getAndAssignLockId(0, 2).value(), 0);
  EXPECT_EQ(generator.isLockIdAssigned(0, 2, 0), true);
  // Out of locks
  EXPECT_EQ(generator.getAndAssignLockId(0, 2), std::nullopt);
  EXPECT_EQ(generator.isLockIdAssigned(0, 2, 16), false);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
