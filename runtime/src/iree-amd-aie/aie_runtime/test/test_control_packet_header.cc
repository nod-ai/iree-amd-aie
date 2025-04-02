// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gtest/gtest.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

namespace mlir::iree_compiler::AMDAIE {

TEST(ControlPacketHeaderTest, OddParity) {
  // Has an even number of 1s, so MSB is set to 1.
  uint32_t word = 0x00000000;
  setOddParityBit(word);
  EXPECT_EQ(word, 0x80000000);
  // Has an odd number of 1s, so MSB remains 0.
  word = 0x00000001;
  setOddParityBit(word);
  EXPECT_EQ(word, 0x00000001);
  // Has an odd number of 1s, so MSB remains 0.
  word = 0x00000002;
  setOddParityBit(word);
  EXPECT_EQ(word, 0x00000002);
  // Has an even number of 1s, so MSB is set to 1.
  word = 0x00000003;
  setOddParityBit(word);
  EXPECT_EQ(word, 0x80000003);
  // Has an odd number of 1s, so MSB remains 0.
  word = 0x10000000;
  setOddParityBit(word);
  EXPECT_EQ(word, 0x10000000);
  // Has an odd number of 1s, so MSB remains 0.
  word = 0x20000000;
  setOddParityBit(word);
  EXPECT_EQ(word, 0x20000000);
  // Has an even number of 1s, so MSB is set to 1.
  word = 0x30000000;
  setOddParityBit(word);
  EXPECT_EQ(word, 0xB0000000);
}

// Parameterized test for device-dependent header construction.
class ControlPacketHeaderTest : public ::testing::TestWithParam<AMDAIEDevice> {
};

TEST_P(ControlPacketHeaderTest, PacketHeader) {
  AMDAIEDevice deviceType = GetParam();
  AMDAIEDeviceModel deviceModel = getDeviceModel(deviceType);

  // Construct the header as:
  // (srcCol << 21) | (srcRow << 16) | (packetType << 12) | (packetId << 0).
  // Then set the odd parity bit at MSB.
  FailureOr<uint32_t> maybeHeader = deviceModel.getPacketHeader(
      /*packetId=*/0, /*packetType=*/0, /*srcRow=*/0, /*srcCol=*/0);
  EXPECT_TRUE(succeeded(maybeHeader));
  EXPECT_EQ(*maybeHeader, 0x80000000);

  maybeHeader = deviceModel.getPacketHeader(
      /*packetId=*/1, /*packetType=*/1, /*srcRow=*/1, /*srcCol=*/1);
  EXPECT_TRUE(succeeded(maybeHeader));
  EXPECT_EQ(*maybeHeader, 0x80211001);

  // Fail because `srcCol` is out of range.
  maybeHeader = deviceModel.getPacketHeader(
      /*packetId=*/0, /*packetType=*/0, /*srcRow=*/0, /*srcCol=*/100);
  EXPECT_TRUE(failed(maybeHeader));

  // Fail because `srcRow` is out of range.
  maybeHeader = deviceModel.getPacketHeader(
      /*packetId=*/0, /*packetType=*/0, /*srcRow=*/100, /*srcCol=*/0);
  EXPECT_TRUE(failed(maybeHeader));

  // Fail because `packetType` is out of range.
  maybeHeader = deviceModel.getPacketHeader(
      /*packetId=*/0, /*packetType=*/100, /*srcRow=*/0, /*srcCol=*/0);
  EXPECT_TRUE(failed(maybeHeader));

  // Fail because `packetId` is out of range.
  maybeHeader = deviceModel.getPacketHeader(
      /*packetId=*/100, /*packetType=*/0, /*srcRow=*/0, /*srcCol=*/0);
  EXPECT_TRUE(failed(maybeHeader));
}

TEST_P(ControlPacketHeaderTest, ControlHeader) {
  AMDAIEDevice deviceType = GetParam();
  AMDAIEDeviceModel deviceModel = getDeviceModel(deviceType);

  // Construct the control header as:
  // (streamId << 24) | (opcode << 22) | ((length - 1) << 20) | (address << 0).
  // Then set the odd parity bit at MSB.
  FailureOr<uint32_t> maybeHeader = deviceModel.getControlHeader(
      /*address=*/0, /*length=*/1, /*opcode=*/0, /*streamId=*/0);
  EXPECT_TRUE(succeeded(maybeHeader));
  EXPECT_EQ(*maybeHeader, 0x80000000);

  maybeHeader = deviceModel.getControlHeader(
      /*address=*/1, /*length=*/1, /*opcode=*/1, /*streamId=*/1);
  EXPECT_TRUE(succeeded(maybeHeader));
  EXPECT_EQ(*maybeHeader, 0x01400001);

  // Fail because `address` is out of range.
  maybeHeader = deviceModel.getControlHeader(
      /*address=*/0x1FFFFF, /*length=*/1, /*opcode=*/0, /*streamId=*/0);
  EXPECT_TRUE(failed(maybeHeader));

  // Fail because `length` is out of range.
  maybeHeader = deviceModel.getControlHeader(
      /*address=*/0, /*length=*/0, /*opcode=*/0, /*streamId=*/0);
  EXPECT_TRUE(failed(maybeHeader));

  // Fail because `length` is out of range.
  maybeHeader = deviceModel.getControlHeader(
      /*address=*/0, /*length=*/100, /*opcode=*/0, /*streamId=*/0);
  EXPECT_TRUE(failed(maybeHeader));

  // Fail because `opcode` is out of range.
  maybeHeader = deviceModel.getControlHeader(
      /*address=*/0, /*length=*/1, /*opcode=*/100, /*streamId=*/0);
  EXPECT_TRUE(failed(maybeHeader));

  // Fail because `streamId` is out of range.
  maybeHeader = deviceModel.getControlHeader(
      /*address=*/0, /*length=*/1, /*opcode=*/0, /*streamId=*/100);
}

INSTANTIATE_TEST_SUITE_P(ControlPacketHeaderTestSuite, ControlPacketHeaderTest,
                         ::testing::Values(AMDAIEDevice::npu1_4col,
                                           AMDAIEDevice::npu4));

}  // namespace mlir::iree_compiler::AMDAIE

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
