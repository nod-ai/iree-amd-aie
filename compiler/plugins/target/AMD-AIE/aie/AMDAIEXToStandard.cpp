// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEXDialect.h"
#include "Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

#define DEBUG_TYPE "amdaiex-standard-lowering"

template <typename MyAIEXOp>
struct AMDAIEXOpRemoval : OpConversionPattern<MyAIEXOp> {
  using OpConversionPattern<MyAIEXOp>::OpConversionPattern;
  using OpAdaptor = typename MyAIEXOp::Adaptor;
  ModuleOp &module;

  AMDAIEXOpRemoval(MLIRContext *context, ModuleOp &m,
                   PatternBenefit benefit = 1)
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
struct AMDAIEXToStandardPass : mlir::OperationPass<mlir::ModuleOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AMDAIEXToStandardPass)

  AMDAIEXToStandardPass()
      : mlir::OperationPass<mlir::ModuleOp>(resolveTypeID()) {}

  llvm::StringRef getArgument() const override {
    return "amdaiex-standard-lowering";
  }

  llvm::StringRef getName() const override { return "AMDAIEXToStandardPass"; }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<AMDAIEXToStandardPass>(
        *static_cast<const AMDAIEXToStandardPass *>(this));
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
    registry.insert<xilinx::AIEX::AIEXDialect>();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet removepatterns(&getContext());
    removepatterns.add<AMDAIEXOpRemoval<NpuDmaMemcpyNdOp>>(m.getContext(), m);
    removepatterns.add<AMDAIEXOpRemoval<NpuDmaWaitOp>>(m.getContext(), m);
    removepatterns.add<AMDAIEXOpRemoval<NpuPushQueueOp>>(m.getContext(), m);
    removepatterns.add<AMDAIEXOpRemoval<NpuWriteRTPOp>>(m.getContext(), m);
    removepatterns.add<AMDAIEXOpRemoval<NpuWrite32Op>>(m.getContext(), m);
    removepatterns.add<AMDAIEXOpRemoval<NpuSyncOp>>(m.getContext(), m);
    removepatterns.add<AMDAIEXOpRemoval<NpuWriteBdOp>>(m.getContext(), m);
    removepatterns.add<AMDAIEXOpRemoval<NpuAddressPatchOp>>(m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(removepatterns))))
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAMDAIEXToStandardPass() {
  return std::make_unique<AMDAIEXToStandardPass>();
}

void registerAMDAIEXToStandardPass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIEXToStandardPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
