// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#define DEBUG_TYPE "iree-amdaie-linalg-function-outlining"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Return true if the strides of `type` make it a contiguous memref.
bool isContiguousMemRef(MemRefType type) {
  ArrayRef<int64_t> shape = type.getShape();
  SmallVector<int64_t, 4> strides;
  int64_t ignoredOffset;
  if (failed(getStridesAndOffset(type, strides, ignoredOffset))) return false;
  int64_t expectedStride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    if (shape[i] == ShapedType::kDynamic) return false;
    if (strides[i] != expectedStride) return false;
    expectedStride *= shape[i];
  }
  return true;
}

/// If `type` is a contiguous memref, return an equivalent memref without any
/// layout attribute. Otherwise, return nullptr.
Type getTypeWithoutLayout(Type type) {
  auto memRefType = dyn_cast<MemRefType>(type);
  if (!memRefType) return {};
  if (!isContiguousMemRef(memRefType)) return {};
  return MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                         MemRefLayoutAttrInterface{},
                         memRefType.getMemorySpace());
}

/// Utility to outline the linalg compute op.
static FailureOr<func::FuncOp> outline(IRRewriter &rewriter, ModuleOp moduleOp,
                                       linalg::LinalgOp computeOp,
                                       const std::string &funcName) {
  // Form outlined FunctionType.
  SmallVector<Type> inputTypes;
  inputTypes.reserve(computeOp->getOperands().size());
  for (const auto &operand : computeOp->getOperands()) {
    Type withoutLayout = getTypeWithoutLayout(operand.getType());
    if (!withoutLayout) {
      return computeOp.emitOpError("has an operand of type ")
             << operand.getType() << " that isn't compatible with outlining.";
    }
    inputTypes.push_back(withoutLayout);
  }
  auto funcType =
      FunctionType::get(rewriter.getContext(), inputTypes, /*outputTypes=*/{});

  // Form outlined FuncSignature.
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto func =
      rewriter.create<func::FuncOp>(moduleOp.getLoc(), funcName, funcType);
  func.setPrivate();

  // Create an entry func block and map the original operands of the compute
  // op to the block arguments.
  Block *funcBody = func.addEntryBlock();
  rewriter.setInsertionPointToStart(funcBody);
  SmallVector<BlockArgument> funcArgs = llvm::map_to_vector(
      func.getArguments(), [&](BlockArgument bbArg) { return bbArg; });
  unsigned bbArgIndex = 0;
  IRMapping operandMap;
  for (Value origOperand : computeOp->getOperands())
    operandMap.map(origOperand, funcArgs[bbArgIndex++]);

  // Clone the compute op while mapping the operand to the function block
  // arguments.
  Operation *clonedComputeOp = rewriter.clone(*computeOp, operandMap);

  // Create terminator op returning the cloned compute op's results.
  rewriter.setInsertionPointToEnd(funcBody);
  rewriter.create<func::ReturnOp>(clonedComputeOp->getLoc(), ValueRange({}));

  return func;
}

/// Utility to check if the linalg op is one we know should not be outlined.
static bool mustNotOutline(linalg::LinalgOp linalgOp) {
  return isa<linalg::CopyOp, linalg::FillOp>(linalgOp);
  // TODO(newling) not all remaining ops should be outlined, not even all
  // remaining matmuls: below some threshold on size (m*n*k) it's not worth
  // outlining (function call overhead). We should extend the blacklist
  // here.
};

class AMDAIELinalgFunctionOutliningPass
    : public impl::AMDAIELinalgFunctionOutliningBase<
          AMDAIELinalgFunctionOutliningPass> {
 public:
  AMDAIELinalgFunctionOutliningPass() = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override;

 private:
  // Used for unique-ifing ID for generating new function names.
  unsigned outlineCounter = 0;

  DenseMap<Operation *, func::FuncOp> computeOpToOutlinedFuncMap;

  static std::string getSpecializedName(linalg::LinalgOp computeOp) {
    // Will result in a function name like `generic_matmul_0_outlined`:
    if (isMatmul(computeOp)) return "_matmul_";
    // Will result in a function name like `generic_elementwise_0_outlined`:
    if (isElementwise(computeOp)) return "_elementwise_";
    // Will result in a function name like `generic_0_outlined`:
    return "_";
  }

  std::string generateFuncName(linalg::LinalgOp computeOp) {
    std::string name = computeOp->getName().stripDialect().str() +
                       getSpecializedName(computeOp) +
                       std::to_string(outlineCounter) + "_outlined";
    ++outlineCounter;
    return name;
  }

  FailureOr<func::FuncOp> retrieveOrCreate(IRRewriter &rewriter,
                                           ModuleOp moduleOp,
                                           linalg::LinalgOp computeOp) {
    // Check if the compute op is equivalent to a previously outlined compute
    // op. If it is, retrieve and return the function generated for the previous
    // compute op.
    for (auto &[op, func] : computeOpToOutlinedFuncMap) {
      if (OperationEquivalence::isEquivalentTo(
              computeOp.getOperation(), op,
              OperationEquivalence::ignoreValueEquivalence, /*flags=*/nullptr,
              OperationEquivalence::IgnoreLocations)) {
        return func;
      }
    }

    std::string funcName = generateFuncName(computeOp);
    while (moduleOp.lookupSymbol(funcName)) {
      funcName = generateFuncName(computeOp);
    }

    FailureOr<func::FuncOp> maybeFunc =
        outline(rewriter, moduleOp, computeOp, funcName);

    if (succeeded(maybeFunc)) {
      computeOpToOutlinedFuncMap[computeOp] = maybeFunc.value();
    }

    return maybeFunc;
  }
};

void AMDAIELinalgFunctionOutliningPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  SmallVector<Operation *> toBeErased;
  WalkResult walkResult = moduleOp.walk([&](linalg::LinalgOp computeOp) {
    if (mustNotOutline(computeOp)) return WalkResult::skip();

    FailureOr<func::FuncOp> maybeFunc =
        retrieveOrCreate(rewriter, moduleOp, computeOp);
    if (failed(maybeFunc)) return WalkResult::interrupt();
    func::FuncOp func = maybeFunc.value();

    // Create a call into the outlined function. The operands of the compute op
    // might need to be cast to a different type to match the outlined function.
    {
      SmallVector<Value> castOperands;
      castOperands.reserve(computeOp->getOperands().size());
      rewriter.setInsertionPoint(computeOp);
      Location loc = computeOp.getLoc();
      for (auto iter : llvm::enumerate(computeOp->getOperands())) {
        Type type = func.getArgumentTypes()[iter.index()];
        Value cast =
            rewriter.createOrFold<memref::CastOp>(loc, type, iter.value());
        castOperands.push_back(cast);
      }
      rewriter.create<func::CallOp>(loc, func, castOperands);
    }

    // We cannot immediately erase the compute op because it'd be used for
    // equivalence check.
    toBeErased.push_back(computeOp);
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) return signalPassFailure();
  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELinalgFunctionOutliningPass() {
  return std::make_unique<AMDAIELinalgFunctionOutliningPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
