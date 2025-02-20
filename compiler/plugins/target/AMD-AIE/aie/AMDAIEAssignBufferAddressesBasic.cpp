// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEDialect.h"
#include "Passes.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "amdaie-assign-buffers-basic"

using namespace mlir;
using namespace xilinx::AIE;

namespace mlir::iree_compiler::AMDAIE {
struct AMDAIEAssignBufferAddressesPassBasic : mlir::OperationPass<DeviceOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AMDAIEAssignBufferAddressesPassBasic)

  AMDAIEAssignBufferAddressesPassBasic()
      : mlir::OperationPass<DeviceOp>(resolveTypeID()) {}

  llvm::StringRef getArgument() const override {
    return "amdaie-assign-buffer-addresses-basic";
  }

  llvm::StringRef getName() const override {
    return "AMDAIEAssignBufferAddressesBasic";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<AMDAIEAssignBufferAddressesPassBasic>(
        *static_cast<const AMDAIEAssignBufferAddressesPassBasic *>(this));
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    int counter = 0;
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      if (!hasName(buffer))
        buffer.setSymName("_anonymous" + std::to_string(counter++));
    });

    DenseMap<TileOp, SetVector<BufferOp>> tileToBuffers;
    device.walk<WalkOrder::PreOrder>([&](BufferOp buffer) {
      tileToBuffers[getTileOp(*buffer)].insert(buffer);
    });

    AMDAIEDeviceModel deviceModel = mlir::iree_compiler::AMDAIE::getDeviceModel(
        static_cast<AMDAIEDevice>(device.getDevice()));
    for (auto [tile, buffers] : tileToBuffers) {
      // Leave room at the bottom of the address range for stack
      int64_t stackRelativeAddress = 0;
      for (auto buffer : buffers) {
        buffer.setStackRelativeAddress(stackRelativeAddress);
        stackRelativeAddress += getAllocationSize(buffer);
      }

      int maxDataMemorySize;
      if (deviceModel.isMemTile(tile.getCol(), tile.getRow()))
        maxDataMemorySize =
            deviceModel.getMemTileSize(tile.getCol(), tile.getRow());
      else
        maxDataMemorySize =
            deviceModel.getLocalMemorySize(tile.getCol(), tile.getRow());
      if (stackRelativeAddress > maxDataMemorySize) {
        InFlightDiagnostic error =
            tile.emitOpError("allocated buffers exceeded available memory (")
            << stackRelativeAddress << ">" << maxDataMemorySize
            << ") even before taking into account the stack!\n";
        return signalPassFailure();
      }
    }
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
createAMDAIEAssignBufferAddressesBasicPass() {
  return std::make_unique<AMDAIEAssignBufferAddressesPassBasic>();
}

void registerAMDAIEAssignBufferAddressesBasic() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIEAssignBufferAddressesBasicPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
