// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-tensor-pad"

namespace mlir::iree_compiler::AMDAIE {

namespace {


class AMDAIETensorPadPass : public AMDAIETensorPadBase<AMDAIETensorPadPass> {
private:
  AMDAIETensorPadOption option = AMDAIETensorPadOption::ParallelDims;

public:
  explicit AMDAIETensorPadPass(AMDAIETensorPadOption option)
      : option(option) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override;
};

void AMDAIETensorPadPass::runOnOperation() {
  llvm::outs() << "BEFORE :-\n" << (*getOperation()) << "\n--------\n";
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  // Preserve the innermost tensor.pad ops (i.e., pad for reduction dims), so we
  // can kick canonicalization patterns to fold outer tensor.pad ops away.
  bool nofold = false;
  utils::IteratorType targetIterType = utils::IteratorType::parallel;
  switch (option) {
  case AMDAIETensorPadOption::ParallelDims:
    LLVM_DEBUG(llvm::dbgs() << "padding parallel dims\n");
    targetIterType = utils::IteratorType::parallel;
    nofold = false;
    break;
  case AMDAIETensorPadOption::ReductionDims:
    LLVM_DEBUG(llvm::dbgs() << "padding reduction dims\n");
    targetIterType = utils::IteratorType::reduction;
    nofold = true;
    break;
  default: // Unreachable.
    assert(false);
    break;
  };
  SmallVector<linalg::LinalgOp> candidates;
  funcOp->walk([&](linalg::LinalgOp op) { 
    if (!isa<linalg::FillOp>(op))
    candidates.push_back(op); 
  });
  for (auto linalgOp : candidates) {
    IRRewriter rewriter(context);
    llvm::outs()<<linalgOp<<"\n";
    llvm::outs()<<"parallel : "<<linalgOp.getNumParallelLoops()<<"\n";
    llvm::outs()<<"reduction : "<<linalgOp.getNumReductionLoops()<<"\n";
    LLVM_DEBUG(llvm::dbgs() << "candidate: " << linalgOp);

    // Early exit if there are no target dimensions to pad.
    if (option == AMDAIETensorPadOption::ParallelDims &&
        linalgOp.getNumParallelLoops() == 0)
      continue;
    if (option == AMDAIETensorPadOption::ReductionDims &&
        linalgOp.getNumReductionLoops() == 0)
      continue;

    IRRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(linalgOp);

    SmallVector<int64_t> paddingDims;
    for (auto [index, iterType] :
         llvm::enumerate(linalgOp.getIteratorTypesArray())) {
      if (iterType == targetIterType) {
        paddingDims.push_back(index);
      }
    }
    llvm::outs()<<"DB - 1\n";

    SmallVector<Attribute> paddingValueAttributes;
    OpBuilder builder(context);
    for (auto &operand : linalgOp->getOpOperands()) {
      auto elemType = getElementTypeOrSelf(operand.get().getType());
      if (auto complexTy = elemType.dyn_cast<ComplexType>()) {
        auto zeroAttr = builder.getZeroAttr(complexTy.getElementType());
        paddingValueAttributes.push_back(
            ArrayAttr::get(context, {zeroAttr, zeroAttr}));
        continue;
      }
      paddingValueAttributes.push_back(builder.getZeroAttr(elemType));
    }
    llvm::outs()<<"DB - 2\n";

    // If nofold is true, we must create pad ops for input operands. The output
    // operands mostly come from scf.for iter_arg. We can not infer the bounding
    // box for such case, so we do not force pad happening.
    SmallVector<bool> noFold(linalgOp.getNumDpsInputs(), nofold);
    noFold.append(linalgOp.getNumDpsInits(), false);
    
    llvm::outs()<<"DB - 3\n";

    auto options =
        linalg::LinalgPaddingOptions()
            .setPaddingDimensions(paddingDims)
            .setPaddingValues(paddingValueAttributes)
            .setPackPaddings(noFold)
            .setCopyBackOp(linalg::LinalgPaddingOptions::CopyBackOp::None);
    
    llvm::outs()<<"DB - 4\n";
    FailureOr<linalg::LinalgOp> maybePaddedLinalgOp =
        linalg::padAndHoistLinalgOp(rewriter, linalgOp, options);
    
    llvm::outs()<<"DB - 5\n";
    if (failed(maybePaddedLinalgOp)) {
      LLVM_DEBUG(llvm::dbgs() << "failed on padding\n");
      return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    context->getLoadedDialect<tensor::TensorDialect>()
        ->getCanonicalizationPatterns(patterns);
    tensor::PadOp::getCanonicalizationPatterns(patterns, context);
    
    llvm::outs()<<"DB - 6\n";
    llvm::outs() << "MID :-\n" << (*getOperation()) << "\n--------\n";
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
      return signalPassFailure();
    }
  }
  llvm::outs() << "AFTER :-\n" << (*getOperation()) << "\n--------\n";
}

} // namespace

std::unique_ptr<OperationPass<>>
createAMDAIETensorPadPass(AMDAIETensorPadOption option) {
  return std::make_unique<AMDAIETensorPadPass>(option);
}
} // namespace mlir::iree_compiler::AMDAIE
