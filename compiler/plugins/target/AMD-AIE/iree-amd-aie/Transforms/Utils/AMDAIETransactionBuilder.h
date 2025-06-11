// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIETRANSACTIONBUILDER_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIETRANSACTIONBUILDER_H_

#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "iree-amd-aie/aie_runtime/iree_aie_configure.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

namespace mlir::iree_compiler::AMDAIE {

/// Generate uController transactions using `aie-rt` APIs.
class TransactionBuilder {
 public:
  AMDAIE::AMDAIEDeviceModel deviceModel;
  TransactionBuilder(AMDAIE::AMDAIEDeviceModel deviceModel)
      : deviceModel(std::move(deviceModel)) {}

  void clearAndInitialize();
  void dumpTransactionAsHex() const;
  size_t getInstructionSize() const;
  ArrayRef<uint32_t> finalizeAndReturnInstructions();

  LogicalResult appendAddressPatch(uint32_t addr, uint32_t argIdx,
                                   uint32_t offset);

  LogicalResult appendDmaStartOp(
        uint32_t col, uint32_t row, uint32_t bdId, uint32_t bufferLength,
        uint32_t bufferOffset, bool enablePacket, uint32_t packetId,
        uint32_t packetType, ArrayRef<int32_t> sizes,
        SmallVector<int32_t> strides, uint32_t iterationCurrent,
        uint32_t iterationSize, uint32_t iterationStride, uint32_t nextBd,
        bool useNextBd, bool validBd, int32_t lockRelVal, uint32_t lockRelId,
        bool lockAcqEnable, int32_t lockAcqVal, uint32_t lockAcqId);

  LogicalResult appendDmaStartOp(Operation* dmaStartOp);

  LogicalResult appendTCTSync(uint32_t col, uint32_t row, uint32_t direction,
                              uint32_t rowNum, uint32_t colNum,
                              uint32_t channel);

  LogicalResult appendPushToQueueOp(uint32_t col, uint32_t row,
                                    AMDAIE::DMAChannelDir direction,
                                    uint32_t channel, uint32_t bdId,
                                    uint32_t repeatCount, bool issueToken);

  LogicalResult appendWriteBdOp(
      uint32_t col, uint32_t row, uint32_t bdId, uint32_t bufferLength,
      uint32_t bufferOffset, bool enablePacket, uint32_t packetId,
      uint32_t packetType, ArrayRef<int32_t> sizes,
      SmallVector<int32_t> strides, uint32_t iterationCurrent,
      uint32_t iterationSize, uint32_t iterationStride, uint32_t nextBd,
      bool useNextBd, bool validBd, int32_t lockRelVal, uint32_t lockRelId,
      bool lockAcqEnable, int32_t lockAcqVal, uint32_t lockAcqId);

 private:
  std::vector<uint32_t> instructions;
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif
