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

/// Return true if the strides of `memrefType` are contiguous.
bool isContiguousMemRef(MemRefType memrefType) {
  ArrayRef<int64_t> shape = memrefType.getShape();
  SmallVector<int64_t, 4> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memrefType, strides, offset))) return false;
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
Type getIdentityLayoutType(Type type) {
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
                                       const std::string &outlineFuncName) {
  // Form outlined FunctionType.
  SmallVector<Type> inputTypes = llvm::map_to_vector(
      computeOp.getDpsInputs(),
      [&](Value v) { return getIdentityLayoutType(v.getType()); });

  for (Value val : computeOp.getDpsInits())
    inputTypes.push_back(getIdentityLayoutType(val.getType()));

  // If any of the input types is not set, return failure.
  if (llvm::any_of(inputTypes, [](Type t) { return !t; }))
    return computeOp.emitOpError(
        "has inputs with types that aren't compatible with outlining");

  auto outlinedFuncType =
      FunctionType::get(rewriter.getContext(), inputTypes, /*outputTypes=*/{});

  // Form outlined FuncSignature
  rewriter.setInsertionPointToStart(moduleOp.getBody());
  auto outlinedFunc = rewriter.create<func::FuncOp>(
      moduleOp.getLoc(), outlineFuncName, outlinedFuncType);
  outlinedFunc.setPrivate();

  // Create an entry func block and map the original operands of the compute
  // op to the block arguments.
  Block *outlinedFuncBody = outlinedFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(outlinedFuncBody);
  SmallVector<BlockArgument> outlinedFuncArgs = llvm::map_to_vector(
      outlinedFunc.getArguments(), [&](BlockArgument bbArg) { return bbArg; });
  unsigned bbArgIndex = 0;
  IRMapping operandMap;
  for (Value origOperand : computeOp.getDpsInputs())
    operandMap.map(origOperand, outlinedFuncArgs[bbArgIndex++]);
  for (Value origOperand : computeOp.getDpsInits())
    operandMap.map(origOperand, outlinedFuncArgs[bbArgIndex++]);

  // Clone the compute op while mapping the operand to the function block
  // arguments.
  Operation *clonedComputeOp = rewriter.clone(*computeOp, operandMap);

  // Create terminator op returning the cloned compute op's results.
  rewriter.setInsertionPointToEnd(outlinedFuncBody);
  rewriter.create<func::ReturnOp>(clonedComputeOp->getLoc(), ValueRange({}));

  return outlinedFunc;
}

/// Utility to check if the linalg op is one we know should not be outlined.
static bool mustNotOutline(linalg::LinalgOp linalgOp) {
  return isa<linalg::CopyOp, linalg::FillOp>(linalgOp);
  // TODO(newling) not all remaining ops should be outlined, not even all
  // remaining matmuls: below some threshold on size (m*n*k) it's not worth
  // outlining (function call overhead).
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
  // Used for unique-ifing the outlined function names.
  unsigned outlineCounter = 0;

  DenseMap<Operation *, func::FuncOp> computeOpToOutlinedFuncMap;

  static std::string getSpecializedName(linalg::LinalgOp computeOp) {
    // Will result in a function name like `generic_matmul_2_outlined`:
    if (isMatmul(computeOp)) return "_matmul_";
    // Will result in a function name like `generic_elementwise_2_outlined`:
    if (isElementwise(computeOp)) return "_elementwise_";
    // Will result in a function name like `generic_2_outlined`:
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
    for (auto &[op, funcOp] : computeOpToOutlinedFuncMap) {
      if (OperationEquivalence::isEquivalentTo(
              computeOp.getOperation(), op,
              OperationEquivalence::ignoreValueEquivalence, /*flags=*/nullptr,
              OperationEquivalence::IgnoreLocations)) {
        return funcOp;
      }
    }

    std::string outlineFuncName = generateFuncName(computeOp);
    while (moduleOp.lookupSymbol(outlineFuncName)) {
      outlineFuncName = generateFuncName(computeOp);
    }

    FailureOr<func::FuncOp> maybeFuncOp =
        outline(rewriter, moduleOp, computeOp, outlineFuncName);

    if (succeeded(maybeFuncOp))
      computeOpToOutlinedFuncMap[computeOp] = maybeFuncOp.value();

    return maybeFuncOp;
  }
};

void AMDAIELinalgFunctionOutliningPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  SmallVector<Operation *> toBeErased;
  WalkResult walkResult = moduleOp.walk([&](linalg::LinalgOp computeOp) {
    if (mustNotOutline(computeOp)) return WalkResult::skip();

    FailureOr<func::FuncOp> maybeFuncOp =
        retrieveOrCreate(rewriter, moduleOp, computeOp);
    if (failed(maybeFuncOp)) return WalkResult::interrupt();
    func::FuncOp outlinedFuncOp = maybeFuncOp.value();

    // Create a call into the outlined function. The operands of the compute op
    // might need to be cast to a different type to match the outlined function.
    {
      SmallVector<Value> castOperands;
      castOperands.reserve(computeOp->getOperands().size());
      rewriter.setInsertionPoint(computeOp);
      Location loc = computeOp.getLoc();
      for (auto iter : llvm::enumerate(computeOp->getOperands())) {
        Type type = outlinedFuncOp.getArgumentTypes()[iter.index()];
        Value cast =
            rewriter.createOrFold<memref::CastOp>(loc, type, iter.value());
        castOperands.push_back(cast);
      }
      rewriter.create<func::CallOp>(loc, outlinedFuncOp, castOperands);
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
