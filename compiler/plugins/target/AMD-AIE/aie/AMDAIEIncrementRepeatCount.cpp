// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEDialect.h"
#include "Passes.h"

#define DEBUG_TYPE "amdaie-increment-repeat-count"

using namespace mlir;
using namespace xilinx::AIE;

namespace mlir::iree_compiler::AMDAIE {
struct AMDAIEIncrementRepeatCountPass
    : public impl::AMDAIEIncrementRepeatCountBase<
          AMDAIEIncrementRepeatCountPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp deviceOp = getOperation();
    OpBuilder builder(deviceOp.getContext());
    deviceOp.walk([&](DMAStartOp dmaStartOp) {
      dmaStartOp->setAttr("repeat_count", builder.getI8IntegerAttr(
                                              dmaStartOp.getRepeatCount() + 1));
    });
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
createAMDAIEIncrementRepeatCountPass() {
  return std::make_unique<AMDAIEIncrementRepeatCountPass>();
}

void registerAMDAIEIncrementRepeatCount() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIEIncrementRepeatCountPass();
  });
}

}  // namespace mlir::iree_compiler::AMDAIE
