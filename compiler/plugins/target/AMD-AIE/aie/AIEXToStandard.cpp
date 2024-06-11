// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Passes.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

#define GEN_PASS_DECL_AIEXTOSTANDARD
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
#undef GEN_PASS_DECL_AIEXTOSTANDARD

#define GEN_PASS_DEF_AIEXTOSTANDARD
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
#undef GEN_PASS_DEF_AIEXTOSTANDARD

template <typename MyAIEXOp>
struct AIEXOpRemoval : OpConversionPattern<MyAIEXOp> {
  using OpConversionPattern<MyAIEXOp>::OpConversionPattern;
  using OpAdaptor = typename MyAIEXOp::Adaptor;
  ModuleOp &module;

  AIEXOpRemoval(MLIRContext *context, ModuleOp &m, PatternBenefit benefit = 1)
      : OpConversionPattern<MyAIEXOp>(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(
      MyAIEXOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Operation *Op = op.getOperation();
    rewriter.eraseOp(Op);
    return success();
  }
};

namespace mlir::iree_compiler::AMDAIE {
struct AIEXToStandardPass : ::impl::AIEXToStandardBase<AIEXToStandardPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet removepatterns(&getContext());
    removepatterns.add<AIEXOpRemoval<NpuDmaMemcpyNdOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuDmaWaitOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuPushQueueOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuWriteRTPOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuWrite32Op>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuSyncOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuWriteBdOp>>(m.getContext(), m);
    removepatterns.add<AIEXOpRemoval<NpuAddressPatchOp>>(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(removepatterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAIEXToStandardPass() {
  return std::make_unique<AIEXToStandardPass>();
}

void registerAIEXToStandardPass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAIEXToStandardPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
