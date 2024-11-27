// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-controlcode-to-half-dma-cpy-nd"

namespace mlir::iree_compiler::AMDAIE {

struct DmaCpyNdToHalfDmaCpyNdConverter final
    : OpConversionPattern<AMDAIE::NpuDmaCpyNdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      AMDAIE::NpuDmaCpyNdOp dmaOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "matchAndRewrite[AMDAIE::NpuDmaCpyNdOp]\n");
    AMDAIE::ConnectionOp connectionOp = dmaOp.getConnectionOp();
    if (!connectionOp) {
      return dmaOp.emitOpError()
             << "should operate on an `amdaie.connection` op";
    }
    // Convert source half.
    Value source =
        dmaOp.getSource() ? dmaOp.getSource() : connectionOp.getSource();
    if (connectionOp.getSourceChannels().size() != 1)
      return connectionOp.emitOpError() << "expected a single source channel";
    auto sourceChannelOp = dyn_cast<AMDAIE::ChannelOp>(
        connectionOp.getSourceChannels()[0].getDefiningOp());
    bool hasAsyncSourceToken =
        llvm::any_of(dmaOp.getAsyncTokens(), [](Value token) {
          return isa<AMDAIE::AsyncSourceTokenType>(token.getType());
        });
    SmallVector<Type> resultTypes = {
        rewriter.getType<AMDAIE::AsyncTokenType>()};
    TypeRange sourceResultTypes =
        hasAsyncSourceToken ? TypeRange{resultTypes} : TypeRange{};
    rewriter.setInsertionPoint(dmaOp);
    auto sourceDma = rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
        dmaOp.getLoc(), sourceResultTypes, connectionOp, source,
        dmaOp.getSourceMixedOffsets(), dmaOp.getSourceMixedSizes(),
        dmaOp.getSourceMixedStrides(), dmaOp.getSourceBdId(), sourceChannelOp);

    // Convert target half.
    Value target =
        dmaOp.getTarget() ? dmaOp.getTarget() : connectionOp.getTarget();
    if (connectionOp.getTargetChannels().size() != 1)
      return connectionOp.emitOpError() << "expected a single target channel";
    auto targetChannelOp = dyn_cast<AMDAIE::ChannelOp>(
        connectionOp.getTargetChannels()[0].getDefiningOp());
    bool hasAsyncTargetToken =
        llvm::any_of(dmaOp.getAsyncTokens(), [](Value token) {
          return isa<AMDAIE::AsyncTargetTokenType>(token.getType());
        });
    TypeRange targetResultTypes =
        hasAsyncTargetToken ? TypeRange{resultTypes} : TypeRange{};
    auto targetDma = rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
        dmaOp.getLoc(), targetResultTypes, connectionOp, target,
        dmaOp.getTargetMixedOffsets(), dmaOp.getTargetMixedSizes(),
        dmaOp.getTargetMixedStrides(), dmaOp.getTargetBdId(), targetChannelOp);
    if (dmaOp.getNumResults() == 1) {
      if (sourceDma.getNumResults() == 1) {
        rewriter.replaceUsesWithIf(
            dmaOp.getResult(0), sourceDma.getResult(0), [&](OpOperand &use) {
              return isa<AMDAIE::AsyncSourceTokenType>(use.get().getType()) &&
                     isa<AMDAIE::NpuDmaWaitOp>(use.getOwner());
            });
      }
      if (targetDma.getNumResults() == 1) {
        rewriter.replaceUsesWithIf(
            dmaOp.getResult(0), targetDma.getResult(0), [&](OpOperand &use) {
              return isa<AMDAIE::AsyncTargetTokenType>(use.get().getType()) &&
                     isa<AMDAIE::NpuDmaWaitOp>(use.getOwner());
            });
      }
      if (!dmaOp.getResult(0).use_empty())
        return dmaOp.emitOpError() << "should not have any uses anymore";
    }
    rewriter.eraseOp(dmaOp);
    return success();
  }
};

namespace {
class AMDAIEControlCodeToHalfDmaCpyNdPass
    : public impl::AMDAIEControlCodeToHalfDmaCpyNdBase<
          AMDAIEControlCodeToHalfDmaCpyNdPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEControlCodeToHalfDmaCpyNdPass::runOnOperation() {
  Operation *parentOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  ConversionTarget conversionTarget(*context);
  conversionTarget.addLegalDialect<AMDAIEDialect>();
  conversionTarget.addIllegalOp<AMDAIE::NpuDmaCpyNdOp>();
  patterns.insert<DmaCpyNdToHalfDmaCpyNdConverter>(context);
  if (failed(applyPartialConversion(parentOp, conversionTarget,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEControlCodeToHalfDmaCpyNdPass() {
  return std::make_unique<AMDAIEControlCodeToHalfDmaCpyNdPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
