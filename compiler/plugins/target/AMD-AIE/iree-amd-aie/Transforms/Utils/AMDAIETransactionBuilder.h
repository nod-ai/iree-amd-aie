// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIETRANSACTIONBUILDER_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIETRANSACTIONBUILDER_H_

#include "iree-amd-aie/IR/AMDAIEOps.h"
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

  LogicalResult appendLockOp(AMDAIE::LockOp lockOp);
  LogicalResult appendDmaStartOp(AMDAIE::DMAStartOp dmaStartOp);

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
      uint32_t packetType, uint32_t outOfOrderBdId, ArrayRef<int32_t> sizes,
      SmallVector<int32_t> strides, uint32_t iterationCurrent,
      uint32_t iterationSize, uint32_t iterationStride, uint32_t nextBd,
      bool useNextBd, bool validBd, int32_t lockRelVal, uint32_t lockRelId,
      bool lockAcqEnable, int32_t lockAcqVal, uint32_t lockAcqId);

 private:
  std::vector<uint32_t> instructions;
};

/// Derive the host-side address-patch table for a serialized uController
/// transaction (the binary produced by `TransactionBuilder`, i.e. the same
/// format `appendAddressPatch` emits into).
///
/// Returns a flat list of (byteOffset, argIdx, argPlus) triples. For each
/// `XAIE_IO_CUSTOM_OP_DDR_PATCH` op, `byteOffset` is the offset (in bytes, from
/// the start of `txn`) of the BD-address word that must be patched at dispatch
/// time with `args[argIdx] + argPlus`. A single BLOCKWRITE may program several
/// consecutive BDs, so each DDR_PATCH is matched to the BLOCKWRITE whose
/// register span [reg, reg + payload_bytes) contains its BD base
/// (`regaddr & ~0xF`), and `byteOffset` points into that BLOCKWRITE's payload.
///
/// This is the single place that parses the serialized transaction binary: the
/// amdxdna HAL's ERT_CMD_CHAIN path cannot rely on firmware-side address
/// patching, so it host-patches BD addresses using this table without itself
/// understanding the transaction format.
std::vector<uint32_t> deriveHostPatchTableFromTransaction(
    ArrayRef<uint32_t> txn);

}  // namespace mlir::iree_compiler::AMDAIE

#endif
