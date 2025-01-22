// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <filesystem>

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-split-control-packet-data"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Splits the data of a control packet into multiple smaller control packets
/// if the data exceeds the maximum allowed size.
struct SplitControlPacketData
    : public OpRewritePattern<AMDAIE::NpuControlPacketOp> {
  using OpRewritePattern::OpRewritePattern;
  uint32_t maxLength;

  SplitControlPacketData(MLIRContext *context, uint32_t maxLength)
      : OpRewritePattern(context), maxLength(maxLength) {}

  LogicalResult matchAndRewrite(AMDAIE::NpuControlPacketOp op,
                                PatternRewriter &rewriter) const override {
    // If below the threshold, no need to split.
    uint32_t length = op.getLength();
    if (length <= maxLength) return failure();

    rewriter.setInsertionPoint(op);
    uint32_t addr = op.getAddress();
    std::optional<ArrayRef<int32_t>> maybeData =
        op.getDataFromArrayOrResource();

    // Split the data into smaller chunks, with each chunk having at most
    // `maxLength` elements.
    for (uint32_t i = 0; i < length; i += maxLength) {
      uint32_t subLength = std::min(maxLength, length - i);
      DenseI32ArrayAttr dataAttr;
      if (maybeData.has_value()) {
        ArrayRef<int32_t> subData = maybeData.value().slice(i, subLength);
        dataAttr = rewriter.getDenseI32ArrayAttr(subData);
      }
      rewriter.create<AMDAIE::NpuControlPacketOp>(
          rewriter.getUnknownLoc(),
          /*address=*/rewriter.getUI32IntegerAttr(addr),
          /*length=*/rewriter.getUI32IntegerAttr(subLength),
          /*opcode=*/rewriter.getUI32IntegerAttr(op.getOpcode()),
          /*stream_id=*/rewriter.getUI32IntegerAttr(op.getStreamId()),
          /*data=*/dataAttr);

      // Update the address for the next control packet.
      addr += subLength * sizeof(int32_t);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class AMDAIESplitControlPacketDataPass
    : public impl::AMDAIESplitControlPacketDataBase<
          AMDAIESplitControlPacketDataPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

void AMDAIESplitControlPacketDataPass::runOnOperation() {
  // Get the maximum length of a control packet.
  Operation *parentOp = getOperation();
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration. This "
           "device-specific information is required to split control packet "
           "operations.";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());
  uint32_t maxLength = deviceModel.getCtrlPktMaxLength();

  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  SplitControlPacketData pattern(context, maxLength);
  patterns.insert<SplitControlPacketData>(std::move(pattern));
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIESplitControlPacketDataPass() {
  return std::make_unique<AMDAIESplitControlPacketDataPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
