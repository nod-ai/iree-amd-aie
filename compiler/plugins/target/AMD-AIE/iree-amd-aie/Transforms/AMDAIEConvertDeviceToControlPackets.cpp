// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <filesystem>

#include "aie/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Target/AMDAIERT.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/aie_runtime/iree_aie_configure.h"
#include "mlir/IR/AsmState.h"

#define DEBUG_TYPE "iree-amdaie-convert-device-to-control-packets"

using Path = std::filesystem::path;

namespace mlir::iree_compiler::AMDAIE {

namespace {
LogicalResult convertDeviceToControlPacket(IRRewriter &rewriter,
                                           xilinx::AIE::DeviceOp deviceOp,
                                           const std::string &pathToElfs) {
  AMDAIEDeviceModel deviceModel = getDeviceModel(deviceOp.getDevice());

  // Start collecting transations.
  TRY_XAIE_API_LOGICAL_RESULT(XAie_StartTransaction, &deviceModel.devInst,
                              XAIE_TRANSACTION_DISABLE_AUTO_FLUSH);
  if (failed(addAllAieElfs(deviceModel, deviceOp, Path{pathToElfs},
                           /*aieSim=*/false))) {
    return failure();
  }
  if (failed(addInitConfig(deviceModel, deviceOp))) return failure();
  if (failed(addAllCoreEnable(deviceModel, deviceOp))) return failure();

  // Export the transactions to a binary buffer.
  uint8_t *txn_ptr =
      XAie_ExportSerializedTransaction(&deviceModel.devInst, 0, 0);
  auto *txn_header = reinterpret_cast<XAie_TxnHeader *>(txn_ptr);
  uint32_t NumOps = txn_header->NumOps;
  txn_ptr += sizeof(XAie_TxnHeader);

  // Create a function named `reconfigure`, with no arguments and no return.
  rewriter.setInsertionPoint(deviceOp);
  auto funcOp =
      rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), "reconfigure",
                                    rewriter.getFunctionType({}, {}));
  Block *funcBody = funcOp.addEntryBlock();
  rewriter.setInsertionPointToEnd(funcBody);
  rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc(), ValueRange({}));
  rewriter.setInsertionPointToStart(funcBody);

  // Create amdaie.workgroup, and insert the control packet operations
  // into its associated control code block.
  auto workgroupOp = rewriter.create<AMDAIE::WorkgroupOp>(funcOp.getLoc());
  rewriter.setInsertionPointToStart(workgroupOp.getBody());
  for (xilinx::AIE::TileOp tileOp : deviceOp.getOps<xilinx::AIE::TileOp>()) {
    auto colIndex = rewriter.create<arith::ConstantIndexOp>(
        rewriter.getUnknownLoc(), tileOp.getCol());
    auto rowIndex = rewriter.create<arith::ConstantIndexOp>(
        rewriter.getUnknownLoc(), tileOp.getRow());
    rewriter.create<AMDAIE::TileOp>(rewriter.getUnknownLoc(), colIndex,
                                    rowIndex);
  }
  Block *controlCodeBlock = workgroupOp.getControlCode().getBody();
  rewriter.setInsertionPointToStart(controlCodeBlock);

  // Masked writes are not natively supported in control packets. To emulate
  // this functionality, we buffer the most recent data written to each
  // specified address.
  DenseMap<uint64_t, uint32_t> emulationBuffer;

  // Set the opcode to `write`, indicating data is written only
  // to the `CTRL` port with no return data expected. The `stream_id` is set to
  // 0, as it is irrelevant in this case.
  CtrlPktOpcode opcode = CtrlPktOpcode::write;
  uint32_t stream_id = 0;

  // ID for DenseI32ResourceElementsAttr.
  uint32_t resource_id = 0;

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
        ArrayRef<int32_t> data(reinterpret_cast<int32_t &>(value));
        rewriter.create<AMDAIE::NpuControlPacketOp>(
            rewriter.getUnknownLoc(), addr,
            /*length=*/1, opcode, stream_id,
            /*data=*/rewriter.getDenseI32ArrayAttr(data));
        emulationBuffer[addr] = value;
        txn_ptr += w_header->Size;
        break;
      }
      case XAie_TxnOpcode::XAIE_IO_BLOCKWRITE: {
        XAie_BlockWrite32Hdr *bw_header =
            reinterpret_cast<XAie_BlockWrite32Hdr *>(txn_ptr);
        uint64_t addr = bw_header->RegOff;
        auto payload =
            reinterpret_cast<int32_t *>(txn_ptr + sizeof(XAie_BlockWrite32Hdr));
        // Calculate the payload length in 32-bit words.
        uint32_t length = (bw_header->Size - sizeof(XAie_BlockWrite32Hdr)) / 4;
        ArrayRef<int32_t> data(payload, length);
        auto dataResourceAttr = DenseI32ResourceElementsAttr::get(
            RankedTensorType::get(data.size(),
                                  IntegerType::get(rewriter.getContext(), 32)),
            "ctrl_pkt_data_" + std::to_string(resource_id++),
            HeapAsmResourceBlob::allocateAndCopyInferAlign(data));
        rewriter.create<AMDAIE::NpuControlPacketOp>(
            rewriter.getUnknownLoc(), addr, length, opcode, stream_id,
            dataResourceAttr);
        // Update the emulation buffer for the whole block of data.
        for (size_t i = 0; i < length; i += 1) {
          emulationBuffer[addr] = reinterpret_cast<uint32_t *>(payload)[i];
          addr += sizeof(uint32_t);
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
        ArrayRef<int32_t> data(reinterpret_cast<int32_t &>(value));
        rewriter.create<AMDAIE::NpuControlPacketOp>(
            rewriter.getUnknownLoc(), addr,
            /*length=*/1, opcode, stream_id,
            rewriter.getDenseI32ArrayAttr(data));
        emulationBuffer[addr] = value;
        txn_ptr += mw_header->Size;
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
  rewriter.eraseOp(deviceOp);

  return success();
}

class AMDAIEConvertDeviceToControlPacketsPass
    : public impl::AMDAIEConvertDeviceToControlPacketsBase<
          AMDAIEConvertDeviceToControlPacketsPass> {
 public:
  AMDAIEConvertDeviceToControlPacketsPass(
      const AMDAIEConvertDeviceToControlPacketsOptions &options)
      : AMDAIEConvertDeviceToControlPacketsBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<AMDAIEDialect, xilinx::AIE::AIEDialect, func::FuncDialect>();
  }

  void runOnOperation() override;
};

void AMDAIEConvertDeviceToControlPacketsPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp->getContext());

  // Make sure there is only one device op.
  SmallVector<xilinx::AIE::DeviceOp> deviceOps;
  parentOp->walk([&](xilinx::AIE::DeviceOp deviceOp) {
    deviceOps.push_back(deviceOp);
    return WalkResult::advance();
  });
  if (deviceOps.size() != 1) {
    parentOp->emitOpError("expected exactly one xilinx.aie.device op");
    return signalPassFailure();
  }

  // Start the conversion.
  if (failed(convertDeviceToControlPacket(rewriter, deviceOps[0], pathToElfs)))
    return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEConvertDeviceToControlPacketsPass(
    AMDAIEConvertDeviceToControlPacketsOptions options) {
  return std::make_unique<AMDAIEConvertDeviceToControlPacketsPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
