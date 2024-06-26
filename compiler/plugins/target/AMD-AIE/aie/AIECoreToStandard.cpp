// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Passes.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::vector;
using namespace xilinx;
using namespace xilinx::AIE;

#define GEN_PASS_DECL_AIECORETOSTANDARD
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
#undef GEN_PASS_DECL_AIECORETOSTANDARD

#define GEN_PASS_DEF_AIECORETOSTANDARD
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
#undef GEN_PASS_DEF_AIECORETOSTANDARD

template <typename MyAIEOp>
struct AIEOpRemoval : OpConversionPattern<MyAIEOp> {
  using OpConversionPattern<MyAIEOp>::OpConversionPattern;
  using OpAdaptor = typename MyAIEOp::Adaptor;
  ModuleOp &module;

  AIEOpRemoval(MLIRContext *context, ModuleOp &m, PatternBenefit benefit = 1)
      : OpConversionPattern<MyAIEOp>(context, benefit), module(m) {}

  LogicalResult matchAndRewrite(
      MyAIEOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct AIEUseLockToStdLowering : OpConversionPattern<UseLockOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;

  AIEUseLockToStdLowering(MLIRContext *context, ModuleOp &m,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(context, benefit), module(m) {}
  LogicalResult matchAndRewrite(
      UseLockOp useLock, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type int32Type = IntegerType::get(rewriter.getContext(), 32);
    if (!isa<DeviceOp>(useLock->getParentOp())) {
      // Generate the intrinsic name
      std::string funcName = "llvm.aie2.";
      if (useLock.acquire() || useLock.acquireGE())
        funcName += "acquire";
      else if (useLock.release())
        funcName += "release";

      func::FuncOp useLockFunc = module.lookupSymbol<func::FuncOp>(funcName);
      if (!useLockFunc) {
        OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointAfter(useLock->getParentOfType<DeviceOp>());
        useLockFunc = rewriter.create<func::FuncOp>(
            rewriter.getUnknownLoc(), funcName,
            FunctionType::get(rewriter.getContext(), {int32Type, int32Type},
                              {}));
        rewriter.restoreInsertionPoint(ip);
        useLockFunc.setPrivate();
      }

      SmallVector<Value, 2> args;
      auto lockValue = useLock.getLockValue();

      // AIE2 acquire greater equal is encoded as a negative value.
      if (useLock.acquireGE()) lockValue = -lockValue;
      args.push_back(rewriter.create<arith::IndexCastOp>(
          useLock.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          useLock.getLock()));
      args.push_back(rewriter.create<arith::ConstantOp>(
          useLock.getLoc(), IntegerType::get(rewriter.getContext(), 32),
          rewriter.getI32IntegerAttr(lockValue)));

      rewriter.create<func::CallOp>(rewriter.getUnknownLoc(), useLockFunc,
                                    args);
    }
    rewriter.eraseOp(useLock);
    return success();
  }
};

struct AIEBufferToStandard : OpConversionPattern<BufferOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;
  int tileCol = 0;
  int tileRow = 0;
  AIEBufferToStandard(MLIRContext *context, ModuleOp &m,
                      PatternBenefit benefit = 1, int tileCol = -1,
                      int tileRow = -1)
      : OpConversionPattern(context, benefit),
        module(m),
        tileCol(tileCol),
        tileRow(tileRow) {}
  LogicalResult matchAndRewrite(
      BufferOp buffer, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPointToStart(module.getBody());
    auto t = llvm::cast<MemRefType>(buffer.getType());
    int col = llvm::cast<TileOp>(buffer.getTile().getDefiningOp()).getCol();
    int row = llvm::cast<TileOp>(buffer.getTile().getDefiningOp()).getRow();
    auto symName = buffer.name().getValue();
    mlir::ElementsAttr initValue = buffer.getInitialValueAttr();
    // Don't emit initialization for cores that don't "own" the buffer (to
    // prevent duplication in the data section of the elf/object file)
    if ((tileRow != row && tileRow != -1) || (tileCol != col && tileCol != -1))
      initValue = nullptr;
    rewriter.create<memref::GlobalOp>(
        rewriter.getUnknownLoc(), symName, rewriter.getStringAttr("public"),
        buffer.getType(), initValue, /*constant*/ false,
        /*alignment*/ nullptr);

    for (auto &use : make_early_inc_range(buffer.getResult().getUses())) {
      Operation *user = use.getOwner();
      rewriter.setInsertionPoint(user);
      auto allocated = rewriter.create<memref::GetGlobalOp>(
          rewriter.getUnknownLoc(), t, symName);
      // Assume that buffers are aligned so they can be vectorized.
      rewriter.create<memref::AssumeAlignmentOp>(rewriter.getUnknownLoc(),
                                                 allocated, 32);

      use.set(allocated.getResult());
    }

    rewriter.eraseOp(buffer);
    return success();
  }
};
struct AIECoreToStandardFunc : OpConversionPattern<CoreOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;
  IRMapping &mapper;
  DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers;
  int tileCol = 0;
  int tileRow = 0;

  AIECoreToStandardFunc(
      MLIRContext *context, ModuleOp &m, IRMapping &mapper,
      DenseMap<Operation *, SmallVector<BufferOp, 4>> &tileToBuffers,
      PatternBenefit benefit = 1, int tileCol = 1, int tileRow = 1)
      : OpConversionPattern(context, benefit),
        module(m),
        mapper(mapper),
        tileToBuffers(tileToBuffers),
        tileCol(tileCol),
        tileRow(tileRow) {}

  LogicalResult matchAndRewrite(
      CoreOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    int col = op.colIndex();
    int row = op.rowIndex();

    // Only pull code for the indicated function
    if ((tileRow != row && tileRow != -1) ||
        (tileCol != col && tileCol != -1)) {
      rewriter.eraseOp(op);
      return success();
    }

    // The parent should be an AIE.device op.
    rewriter.setInsertionPointAfter(op->getParentOp());

    std::string coreName("core_" + std::to_string(col) + "_" +
                         std::to_string(row));
    auto coreFunc = rewriter.create<func::FuncOp>(
        rewriter.getUnknownLoc(), coreName,
        FunctionType::get(rewriter.getContext(), {}, {}));

    rewriter.cloneRegionBefore(op.getBody(), coreFunc.getBody(),
                               coreFunc.getBody().begin(), mapper);

    // Rewrite the AIE.end() op
    coreFunc.getBody().walk([&](Operation *childOp) {
      rewriter.setInsertionPointAfter(childOp);

      if (isa<EndOp>(childOp)) {
        rewriter.create<func::ReturnOp>(rewriter.getUnknownLoc(),
                                        ValueRange({}));
        rewriter.eraseOp(childOp);
      }
    });

    rewriter.eraseOp(op);
    return success();
  }
};

// Move all the ops with OpTy inside device, to just before the device.
template <typename OpTy>
void outlineOps(DeviceOp device) {
  SmallVector<OpTy, 16> ops;
  for (const auto &op : device.getOps<OpTy>()) ops.push_back(op);

  for (const auto &op : ops) op->moveBefore(device);
}

namespace mlir::iree_compiler::AMDAIE {
struct AIECoreToStandardPass
    : ::impl::AIECoreToStandardBase<AIECoreToStandardPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());

    if (m.getOps<DeviceOp>().empty()) {
      m.emitOpError("expected AIE.device operation at toplevel");
      return signalPassFailure();
    }

    // Ensure that we don't have an incorrect target triple.  This may override
    // some bogus target triple in the original mlir.
    m->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
               builder.getStringAttr("aie2"));

    DenseMap<Operation *, SmallVector<BufferOp, 4>> tileToBuffers;

    IRMapping mapper;
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<VectorDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalOp<func::FuncOp, ModuleOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<AIEUseLockToStdLowering>(m.getContext(), m);
    patterns.add<AIEBufferToStandard>(m.getContext(), m, /*benefit*/ 1, tileCol,
                                      tileRow);
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      return signalPassFailure();

    RewritePatternSet outlinePatterns(&getContext());
    outlinePatterns.add<AIECoreToStandardFunc>(m.getContext(), m, mapper,
                                               tileToBuffers, /*benefit*/ 1,
                                               tileCol, tileRow);
    if (failed(applyPartialConversion(m, target, std::move(outlinePatterns))))
      return signalPassFailure();

    // Move all the func.func ops and memref.globals from the device to the
    // module
    DeviceOp device = *m.getOps<DeviceOp>().begin();
    outlineOps<memref::GlobalOp>(device);
    outlineOps<func::FuncOp>(device);

    RewritePatternSet removepatterns(&getContext());
    removepatterns.add<
        AIEOpRemoval<DeviceOp>, AIEOpRemoval<TileOp>, AIEOpRemoval<FlowOp>,
        AIEOpRemoval<MemOp>, AIEOpRemoval<ShimDMAOp>, AIEOpRemoval<ShimMuxOp>,
        AIEOpRemoval<SwitchboxOp>, AIEOpRemoval<LockOp>, AIEOpRemoval<BufferOp>,
        AIEOpRemoval<ExternalBufferOp>, AIEOpRemoval<ShimDMAAllocationOp>,
        AIEOpRemoval<CascadeFlowOp>, AIEOpRemoval<ConfigureCascadeOp>>(
        m.getContext(), m);

    if (failed(applyPartialConversion(m, target, std::move(removepatterns))))
      return signalPassFailure();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAIECoreToStandardPass() {
  return std::make_unique<AIECoreToStandardPass>();
}

void registerAIECoreToStandard() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAIECoreToStandardPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
