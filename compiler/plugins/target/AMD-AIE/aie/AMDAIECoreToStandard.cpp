// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEDialect.h"
#include "Passes.h"
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
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "amdaie-standard-lowering"

using namespace mlir;
using namespace mlir::vector;
using namespace xilinx;
using namespace xilinx::AIE;

struct AMDAIEUseLockToStdLowering : OpConversionPattern<UseLockOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      UseLockOp useLock, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isa<DeviceOp>(useLock->getParentOp())) {
      // Generate the intrinsic name
      std::string funcName = "llvm.aie2.";
      if (useLock.getAction() == LockAction::Acquire ||
          useLock.getAction() == LockAction::AcquireGreaterEqual)
        funcName += "acquire";
      else if (useLock.getAction() == LockAction::Release)
        funcName += "release";
      // TODO(max): this can be simplified with
      // SymbolTable::lookupNearestSymbolFrom if DeviceOp ceases to be a
      // SymbolTable
      func::FuncOp useLockFunc =
          useLock->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
              funcName);

      SmallVector<Value, 2> args;
      int lockValue = useLock.getValue().value_or(1);

      // AIE2 acquire greater equal is encoded as a negative value.
      if (useLock.getAction() == LockAction::AcquireGreaterEqual)
        lockValue = -lockValue;
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

struct AMDAIEBufferToStandard : OpConversionPattern<BufferOp> {
  using OpConversionPattern::OpConversionPattern;
  ModuleOp &module;
  // TODO(max): these should be optionals instead of checking against -1
  // but the pass itself needs to be updated.
  int tileCol = 0;
  int tileRow = 0;
  AMDAIEBufferToStandard(MLIRContext *context, ModuleOp &m, int tileCol = -1,
                         int tileRow = -1)
      : OpConversionPattern(context),
        module(m),
        tileCol(tileCol),
        tileRow(tileRow) {}
  LogicalResult matchAndRewrite(
      BufferOp buffer, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPointToStart(module.getBody());
    auto t = llvm::cast<MemRefType>(buffer.getType());
    StringRef symName = name(buffer).getValue();
    // Don't emit initialization for cores that don't "own" the buffer (to
    // prevent duplication in the data section of the elf/object file)
    rewriter.create<memref::GlobalOp>(
        rewriter.getUnknownLoc(), symName, rewriter.getStringAttr("public"),
        buffer.getType(), nullptr, /*constant*/ false,
        /*alignment*/ nullptr);

    for (OpOperand &use : make_early_inc_range(buffer.getResult().getUses())) {
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

struct AMDAIECoreToStandardFunc : OpConversionPattern<CoreOp> {
  using OpConversionPattern::OpConversionPattern;
  IRMapping &mapper;
  // TODO(max): these should be optionals instead of checking against -1
  // but the pass itself needs to be updated.
  int tileCol = 0;
  int tileRow = 0;

  AMDAIECoreToStandardFunc(MLIRContext *context, IRMapping &mapper,
                           int tileCol = 1, int tileRow = 1)
      : OpConversionPattern(context),
        mapper(mapper),
        tileCol(tileCol),
        tileRow(tileRow) {}

  LogicalResult matchAndRewrite(
      CoreOp coreOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TileOp t = getTileOp(*coreOp);
    int col = t.getCol();
    int row = t.getRow();

    // Only pull code for the indicated function
    if ((tileRow != row && tileRow != -1) ||
        (tileCol != col && tileCol != -1)) {
      rewriter.eraseOp(coreOp);
      return success();
    }

    // The parent should be an AIE.device op.
    rewriter.setInsertionPointAfter(coreOp->getParentOp());

    std::string coreName("core_" + std::to_string(col) + "_" +
                         std::to_string(row));
    auto coreFunc = rewriter.create<func::FuncOp>(
        rewriter.getUnknownLoc(), coreName,
        FunctionType::get(rewriter.getContext(), {}, {}));



    rewriter.cloneRegionBefore(coreOp.getBody(), coreFunc.getBody(),
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

    rewriter.eraseOp(coreOp);
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
struct AMDAIECoreToStandardPass : mlir::OperationPass<ModuleOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AMDAIECoreToStandardPass)

  AMDAIECoreToStandardPass() : mlir::OperationPass<ModuleOp>(resolveTypeID()) {}
  AMDAIECoreToStandardPass(const AMDAIECoreToStandardPass &other)
      : mlir::OperationPass<mlir::ModuleOp>(other) {}

  llvm::StringRef getArgument() const override {
    return "amdaie-standard-lowering";
  }

  llvm::StringRef getName() const override {
    return "AMDAIECoreToStandardPass";
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<AMDAIECoreToStandardPass>(
        *static_cast<const AMDAIECoreToStandardPass *>(this));
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<xilinx::AIE::AIEDialect>();
  }

  mlir::Pass::Option<unsigned> tileCol{
      *this, "tilecol",
      llvm::cl::desc("X coordinate of tile to generate code for"),
      llvm::cl::init(-1)};
  mlir::Pass::Option<unsigned> tileRow{
      *this, "tilerow",
      llvm::cl::desc("Y coordinate of tile to generate code for"),
      llvm::cl::init(-1)};

  void runOnOperation() override {
    ModuleOp m = getOperation();


    if (m.getOps<DeviceOp>().empty()) {
      m.emitOpError("expected AIE.device operation at toplevel");
      return signalPassFailure();
    }

    OpBuilder builder = OpBuilder::atBlockEnd(m.getBody());


    // Ensure that we don't have an incorrect target triple.  This may override
    // some bogus target triple in the original mlir.
    m->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
               builder.getStringAttr("aie2"));

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

    StringAttr privateSym = StringAttr::get(&getContext(), "private");
    auto buildDecl = [&](const std::string &funcName) {
      builder.create<func::FuncOp>(
          builder.getUnknownLoc(), funcName,
          FunctionType::get(builder.getContext(),
                            {builder.getI32Type(), builder.getI32Type()}, {}),
          privateSym, ArrayAttr{}, ArrayAttr{});
    };
    buildDecl("llvm.aie2.acquire");
    buildDecl("llvm.aie2.release");

    patterns.add<AMDAIEUseLockToStdLowering>(m.getContext());
    patterns.add<AMDAIEBufferToStandard>(m.getContext(), m, tileCol, tileRow);
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      return signalPassFailure();

    // Assert that cores are isolated
    {
      SmallVector<CoreOp> coreOps;
      m->walk([&](CoreOp coreOp) { coreOps.push_back(coreOp); });
      for (CoreOp coreOp : coreOps) {
        auto walkResult = coreOp->walk([&](Operation *childOp) {
          if (childOp == coreOp) return WalkResult::advance();
          for (Value operand : childOp->getOperands()) {
            if (Operation *operandOp = operand.getDefiningOp()) {
              if (!coreOp->isAncestor(operandOp)) {
                operandOp->emitOpError(
                    "is not in the core in which it is used. Cores must be "
                    "`isolated` before this point.");
                return WalkResult::interrupt();
              }
            }
          }
          return WalkResult::advance();
        });
        if (walkResult.wasInterrupted()) return signalPassFailure();
      }
    }
    RewritePatternSet outlinePatterns(&getContext());
    outlinePatterns.add<AMDAIECoreToStandardFunc>(m.getContext(), mapper,
                                                  tileCol, tileRow);
    if (failed(applyPartialConversion(m, target, std::move(outlinePatterns))))
      return signalPassFailure();

    // Move all the func.func ops and memref.globals from the device to the
    // module
    DeviceOp device = *m.getOps<DeviceOp>().begin();
    outlineOps<memref::GlobalOp>(device);
    outlineOps<func::FuncOp>(device);

    MLIRContext &context = getContext();
    IRRewriter rewriter(&context);
    rewriter.eraseOp(device);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAMDAIECoreToStandardPass() {
  return std::make_unique<AMDAIECoreToStandardPass>();
}

void registerAMDAIECoreToStandard() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAMDAIECoreToStandardPass();
  });
}
}  // namespace mlir::iree_compiler::AMDAIE
