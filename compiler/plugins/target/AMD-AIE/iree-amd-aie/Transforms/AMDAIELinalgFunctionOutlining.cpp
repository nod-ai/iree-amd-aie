// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
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

/// Utility to outline the linalg compute op.
static FailureOr<func::FuncOp> outline(IRRewriter &rewriter, ModuleOp moduleOp,
                                       linalg::LinalgOp computeOp,
                                       const std::string &funcName) {
  // Form outlined FunctionType.
  for (const auto &operand : computeOp->getOperands()) {
    // Function signatures where the memrefs have layouts (strides / offsets)
    // do not lower from the func dialect to the llvm dialect. So for now,
    // we just do not try to outline ops with operands with non-identity
    // layouts, because the resulting functions won't lower to LLVM.
    // identity layouts:
    // clang-format off
    // https://github.com/llvm/llvm-project/blob/6b0785390d02193d81d8db7fb12279ffa4651afe/mlir/include/mlir/IR/BuiltinAttributeInterfaces.td#L475
    // clang-format on
    if (auto type = dyn_cast<MemRefType>(operand.getType())) {
      MemRefLayoutAttrInterface layout = type.getLayout();
      assert(layout &&
             "MemRefType layout attribute interface should always be present");
      if (!layout.isIdentity()) return failure();
    }
  }
  auto funcType = FunctionType::get(
      rewriter.getContext(), computeOp->getOperandTypes(), /*outputTypes=*/{});

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

/// Utility to check whether the linalg should be outlined if the balanced
/// strategy is enabled.
static bool mustOutlineBalanced(linalg::LinalgOp linalgOp) {
  if (isa<linalg::CopyOp, linalg::FillOp>(linalgOp)) return false;
  if (isElementwise(linalgOp)) return false;
  // TODO(newling) not all remaining ops should be outlined, not even all
  // remaining matmuls: below some threshold on size (m*n*k) it's not worth
  // outlining (function call overhead). We should extend the set of ops that
  // are not outlined here.

  return true;
};

class AMDAIELinalgFunctionOutliningPass
    : public impl::AMDAIELinalgFunctionOutliningBase<
          AMDAIELinalgFunctionOutliningPass> {
 public:
  AMDAIELinalgFunctionOutliningPass() = default;
  AMDAIELinalgFunctionOutliningPass(const AMDAIELinalgFunctionOutliningPass &) {
  }
  AMDAIELinalgFunctionOutliningPass(
      const AMDAIELinalgFunctionOutliningOptions &opts)
      : AMDAIELinalgFunctionOutliningBase(opts) {}

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

  if (outliningStrategy == OutliningStrategy::None) {
    if (emptyFunctions) {
      moduleOp.emitWarning()
          << "The option to empty outlined functions is enabled while the "
             "outlining strategy specifies to not outline any functions, so no "
             "transformation will happen. This combination might not result in "
             "the intended behaviour.";
    }
    return;
  }

  SmallVector<Operation *> toBeErased;
  moduleOp.walk([&](linalg::LinalgOp computeOp) {
    if (outliningStrategy == OutliningStrategy::Balanced &&
        !mustOutlineBalanced(computeOp)) {
      return WalkResult::skip();
    }

    FailureOr<func::FuncOp> maybeFunc =
        retrieveOrCreate(rewriter, moduleOp, computeOp);
    if (failed(maybeFunc)) return WalkResult::interrupt();
    func::FuncOp func = maybeFunc.value();

    rewriter.setInsertionPoint(computeOp);
    rewriter.create<func::CallOp>(computeOp.getLoc(), func,
                                  computeOp->getOperands());

    // We cannot immediately erase the compute op because it'd be used for
    // equivalence check.
    toBeErased.push_back(computeOp);
    return WalkResult::advance();
  });
  for (Operation *op : toBeErased) {
    op->dropAllUses();
    rewriter.eraseOp(op);
  }

  // If the option is set to true, make the body of all outlined functions
  // empty, so that only the return remains. This option to 'do no compute'
  // is useful for benchmarking purposes.
  if (emptyFunctions) {
    for (auto &nameAndFuncOp : computeOpToOutlinedFuncMap) {
      Region &region = nameAndFuncOp.second.getBody();
      Block &block = region.front();
      uint64_t nOperations = block.getOperations().size();
      assert(nOperations > 0 && "expected terminator");
      for (uint64_t i = 0; i < nOperations - 1; ++i) {
        Operation *frontOp = &block.front();
        rewriter.eraseOp(frontOp);
      }
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELinalgFunctionOutliningPass(
    AMDAIELinalgFunctionOutliningOptions options) {
  return std::make_unique<AMDAIELinalgFunctionOutliningPass>(options);
}
}  // namespace mlir::iree_compiler::AMDAIE
