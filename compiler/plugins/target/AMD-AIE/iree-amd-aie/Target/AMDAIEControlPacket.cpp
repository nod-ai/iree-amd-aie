// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AMDAIEControlPacket.h"

#include <filesystem>

#include "AMDAIERT.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#define DEBUG_TYPE "iree-amdaie-target-controlpacket"

using namespace mlir;
using Path = std::filesystem::path;

namespace mlir::iree_compiler::AMDAIE {

LogicalResult convertAieToControlPacket(ModuleOp moduleOp,
                                        xilinx::AIE::DeviceOp deviceOp,
                                        const std::string &outputMlir,
                                        const std::string &tempDir) {
  AMDAIEDeviceModel deviceModel = getDeviceModel(deviceOp.getDevice());

  // Start collecting transations.
  TRY_XAIE_API_LOGICAL_RESULT(XAie_StartTransaction, &deviceModel.devInst,
                              XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
  if (failed(addAllAieElfs(deviceModel, deviceOp, Path{tempDir},
                           /*aieSim=*/false)))
    return failure();
  if (failed(addInitConfig(deviceModel, deviceOp))) return failure();
  if (failed(addAllCoreEnable(deviceModel, deviceOp))) return failure();

  // Export the transactions to a binary buffer.
  uint8_t *txn_ptr =
      XAie_ExportSerializedTransaction(&deviceModel.devInst, 0, 0);
  auto *txn_header = reinterpret_cast<XAie_TxnHeader *>(txn_ptr);
  uint32_t NumOps = txn_header->NumOps;
  txn_ptr += sizeof(XAie_TxnHeader);

  // Create a new MLIR module.
  MLIRContext context;
  context.loadDialect<AMDAIEDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<IREE::HAL::HALDialect>();
  OpBuilder builder(&context);
  ModuleOp newModuleOp = ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(newModuleOp.getBody());

  // Copy the target attributes from the original module.
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(moduleOp);
  newModuleOp->setAttr("hal.executable.target", targetAttr);

  // Create a function named `reconfigure`, with no arguments and no return.
  auto funcOp = builder.create<func::FuncOp>(
      builder.getUnknownLoc(), "reconfigure", builder.getFunctionType({}, {}));
  Block *funcBody = funcOp.addEntryBlock();
  builder.setInsertionPointToEnd(funcBody);
  builder.create<func::ReturnOp>(builder.getUnknownLoc(), ValueRange({}));
  builder.setInsertionPointToStart(funcBody);

  // Create amdaie.workgroup, and insert the control packet operations
  // into its associated control code block.
  auto workgroupOp = builder.create<AMDAIE::WorkgroupOp>(funcOp.getLoc());
  Block *controlCodeBlock = workgroupOp.getControlCode().getBody();
  builder.setInsertionPointToStart(controlCodeBlock);

  // Masked writes are not natively supported in control packets. To emulate
  // this functionality, we buffer the most recent data written to each
  // specified address.
  DenseMap<uint64_t, uint32_t> emulationBuffer;

  // Set `opcode` and `stream_id` to 0 for writing to the NPU.
  uint32_t opcode = 0;
  uint32_t stream_id = 0;

  // Process each operation in the transaction.
  for (uint32_t i = 0; i < NumOps; i++) {
    XAie_OpHdr *op_header = (XAie_OpHdr *)txn_ptr;
    auto opCode = static_cast<AMDAIE::XAie_TxnOpcode>(op_header->Op);
    switch (opCode) {
      case XAie_TxnOpcode::XAIE_IO_WRITE: {
        XAie_Write32Hdr *w_header =
            reinterpret_cast<XAie_Write32Hdr *>(txn_ptr);
        uint64_t addr = w_header->RegOff;
        uint32_t value = w_header->Value;
        emulationBuffer[addr] = value;
        txn_ptr += w_header->Size;
        builder.create<AMDAIE::NpuControlPacketOp>(
            builder.getUnknownLoc(),
            /*address=*/builder.getUI32IntegerAttr(addr),
            /*length=*/builder.getUI32IntegerAttr(1),
            /*opcode=*/builder.getUI32IntegerAttr(opcode),
            /*stream_id=*/builder.getUI32IntegerAttr(stream_id),
            /*data=*/builder.getDenseI32ArrayAttr(ArrayRef<int32_t>(value)));
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_BLOCKWRITE: {
        XAie_BlockWrite32Hdr *bw_header =
            reinterpret_cast<XAie_BlockWrite32Hdr *>(txn_ptr);
        uint64_t addr = bw_header->RegOff;
        auto payload = reinterpret_cast<uint32_t *>(
            txn_ptr + sizeof(XAie_BlockWrite32Hdr));
        // Payload length in 32-bit words.
        uint32_t length = (bw_header->Size - sizeof(XAie_BlockWrite32Hdr)) / 4;
        SmallVector<int32_t> data(payload, payload + length);
        builder.create<AMDAIE::NpuControlPacketOp>(
            builder.getUnknownLoc(),
            /*address=*/builder.getUI32IntegerAttr(addr),
            /*length=*/builder.getUI32IntegerAttr(length),
            /*opcode=*/builder.getUI32IntegerAttr(opcode),
            /*stream_id=*/builder.getUI32IntegerAttr(stream_id),
            /*data=*/builder.getDenseI32ArrayAttr(data));
        // Update the emulation buffer for the whole block of data.
        for (size_t i = 0; i < length; i += 1) {
          emulationBuffer[addr] = data[i];
          addr += sizeof(int32_t);
        }
        txn_ptr += bw_header->Size;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_MASKWRITE: {
        XAie_MaskWrite32Hdr *mw_header =
            reinterpret_cast<XAie_MaskWrite32Hdr *>(txn_ptr);
        uint64_t addr = mw_header->RegOff;
        uint32_t value = mw_header->Value;
        uint32_t mask = mw_header->Mask;
        // Apply the mask to the value and update the emulation buffer.
        if (emulationBuffer.count(addr))
          value = (emulationBuffer[addr] & ~mask) | (value & mask);
        emulationBuffer[addr] = value;
        txn_ptr += mw_header->Size;
        builder.create<AMDAIE::NpuControlPacketOp>(
            builder.getUnknownLoc(),
            /*address=*/builder.getUI32IntegerAttr(addr),
            /*length=*/builder.getUI32IntegerAttr(1),
            /*opcode=*/builder.getUI32IntegerAttr(opcode),
            /*stream_id=*/builder.getUI32IntegerAttr(stream_id),
            /*data=*/builder.getDenseI32ArrayAttr(ArrayRef<int32_t>(value)));
        break;
      }
      default: {
        return deviceOp.emitOpError()
               << "Unsupported opcode in transaction: " << uint8_t(opCode);
      }
    }
  }

  // Clear the transaction.
  free(txn_header);
  TRY_XAIE_API_LOGICAL_RESULT(XAie_ClearTransaction, &deviceModel.devInst);

  // Dump the new MLIR module to a file.
  std::error_code ec;
  llvm::raw_fd_ostream outputFile(outputMlir, ec);
  if (ec) {
    return deviceOp.emitOpError()
           << "Failed to open output file: " << ec.message();
  }
  newModuleOp->print(outputFile);
  outputFile.close();

  return success();
}

}  // namespace mlir::iree_compiler::AMDAIE
