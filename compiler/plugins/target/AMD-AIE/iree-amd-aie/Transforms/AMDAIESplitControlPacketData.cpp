// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <filesystem>

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
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

  LogicalResult matchAndRewrite(AMDAIE::NpuControlPacketOp op,
                                PatternRewriter &rewriter) const override {
    // The maximum length (beats) of a control packet is determined by the
    // number of bits allocated to the `beat` field in the control packet
    // header.
    unsigned maxLength =
        2 << (static_cast<uint8_t>(AMDAIECtrlPktHeader::OPERATION_SHIFT) -
              static_cast<uint8_t>(AMDAIECtrlPktHeader::BEAT_SHIFT));

    uint32_t length = op.getLength();
    // Below the threshold, no need to split.
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

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<SplitControlPacketData>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIESplitControlPacketDataPass() {
  return std::make_unique<AMDAIESplitControlPacketDataPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
