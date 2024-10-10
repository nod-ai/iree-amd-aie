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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "amdaie-standard-lowering"

using namespace mlir;
using namespace xilinx::AIE;

static void lockToStd(UseLockOp useLock, IRRewriter &rewriter) {
  if (!isa<DeviceOp>(useLock->getParentOp())) {
    std::string funcName = [&]() {
      switch (useLock.getAction()) {
        case LockAction::Acquire:
        case LockAction::AcquireGreaterEqual:
          return "llvm.aie2.acquire";
        case LockAction::Release:
          return "llvm.aie2.release";
        default:
          assert(false && "Unknown lock action");
      }
    }();

    // TODO(max): this can be simplified with
    // SymbolTable::lookupNearestSymbolFrom if DeviceOp ceases to be a
    // SymbolTable
    ModuleOp modOp = useLock->getParentOfType<ModuleOp>();
    func::FuncOp func = modOp.lookupSymbol<func::FuncOp>(funcName);

    int lockValue = useLock.getValue().value_or(1);

    // AIE2 acquire greater equal is encoded as a negative value.
    if (useLock.getAction() == LockAction::AcquireGreaterEqual)
      lockValue = -lockValue;

    rewriter.setInsertionPoint(useLock);
    IntegerAttr lockAttr = rewriter.getI32IntegerAttr(lockValue);
    IntegerType type = IntegerType::get(rewriter.getContext(), 32);
    Location loc = useLock.getLoc();

    SmallVector<Value, 2> args{
        rewriter.create<arith::IndexCastOp>(loc, type, useLock.getLock()),
        rewriter.create<arith::ConstantOp>(loc, type, lockAttr)};

    rewriter.create<func::CallOp>(loc, func, args);
  }

  rewriter.eraseOp(useLock);
}

static void bufferToStd(ModuleOp module, BufferOp buffer,
                        IRRewriter &rewriter) {
  Location loc = buffer.getLoc();
  rewriter.setInsertionPointToStart(module.getBody());
  StringRef symName = name(buffer).getValue();
  MemRefType type = llvm::cast<MemRefType>(buffer.getType());
  // Don't emit initialization for cores that don't "own" the buffer (to
  // prevent duplication in the data section of the elf/object file)
  rewriter.create<memref::GlobalOp>(
      loc, symName, rewriter.getStringAttr("public"), type, nullptr,
      /*constant*/ false,
      /*alignment*/ nullptr);

  for (OpOperand &use : make_early_inc_range(buffer.getResult().getUses())) {
    Operation *user = use.getOwner();
    rewriter.setInsertionPoint(user);

    auto allocated = rewriter.create<memref::GetGlobalOp>(loc, type, symName);
    // Assume that buffers are aligned so they can be vectorized.
    rewriter.create<memref::AssumeAlignmentOp>(loc, allocated, 32);
    use.set(allocated.getResult());
  }

  rewriter.eraseOp(buffer);
}

static void coreToStd(CoreOp coreOp, IRRewriter &rewriter, int tileCol,
                      int tileRow) {
  TileOp t = getTileOp(*coreOp);
  int col = t.getCol();
  int row = t.getRow();

  // Only pull code for the indicated function
  if ((tileRow != row && tileRow != -1) || (tileCol != col && tileCol != -1)) {
    rewriter.eraseOp(coreOp);
    return;
  }

  // The parent should be an AIE.device op.
  rewriter.setInsertionPointAfter(coreOp->getParentOp());

  // LLVM-style of the above (creating a string attribute):
  std::string fName = "core_" + std::to_string(col) + "_" + std::to_string(row);
  auto coreFunc = rewriter.create<func::FuncOp>(
      rewriter.getUnknownLoc(), fName,
      FunctionType::get(rewriter.getContext(), {}, {}));

  IRMapping mapper;
  rewriter.cloneRegionBefore(coreOp.getBody(), coreFunc.getBody(),
                             coreFunc.getBody().begin(), mapper);

  // Rewrite the AIE.end op
  coreFunc.getBody().walk([&](EndOp endOp) {
    rewriter.setInsertionPointAfter(endOp);
    rewriter.create<func::ReturnOp>(endOp->getLoc(), ValueRange({}));
    rewriter.eraseOp(endOp);
  });

  rewriter.eraseOp(coreOp);
}

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

  // Assert that cores are isolated
  static bool coresAreIsolated(ModuleOp m) {
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
      if (walkResult.wasInterrupted()) return false;
    }
    return true;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();

    if (m.getOps<DeviceOp>().empty()) {
      m.emitOpError("expected AIE.device operation at toplevel");
      return signalPassFailure();
    }

    MLIRContext *ctx = &getContext();
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPointToEnd(m.getBody());

    // Ensure that we don't have an incorrect target triple. This may override
    // some bogus target triple in the original mlir.
    m->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
               rewriter.getStringAttr("aie2"));

    StringAttr privateSym = StringAttr::get(ctx, "private");
    auto buildDecl = [&](const std::string &funcName) {
      rewriter.create<func::FuncOp>(
          rewriter.getUnknownLoc(), funcName,
          FunctionType::get(ctx, {rewriter.getI32Type(), rewriter.getI32Type()},
                            {}),
          privateSym, ArrayAttr{}, ArrayAttr{});
    };
    buildDecl("llvm.aie2.acquire");
    buildDecl("llvm.aie2.release");

    m.walk([&](UseLockOp useLock) { lockToStd(useLock, rewriter); });

    m.walk([&](BufferOp buffer) { bufferToStd(m, buffer, rewriter); });

    if (!coresAreIsolated(m)) return signalPassFailure();

    m.walk(
        [&](CoreOp coreOp) { coreToStd(coreOp, rewriter, tileCol, tileRow); });

    // Move all the func.func ops and memref.globals from device to module.
    DeviceOp device = *m.getOps<DeviceOp>().begin();
    outlineOps<memref::GlobalOp>(device);
    outlineOps<func::FuncOp>(device);
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
