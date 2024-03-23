// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// IREE's GenericVectorization pass has many options, patterns, and vectorizes
// some non-linalg ops (tensor.pad). It has logic to choose vector sizes. This
// AIE-specific pass is a minimal version of GenericVectorization tailored to
// the needs of amd-aie. iree-amd-aie uses linalg.copy ops to move data between
// memory spaces (DDR, memory tile, core). These copy ops should not be
// vectorized to vector transfer_read/transfer_write ops. This 'fork' of
// GenericVectorization will be extended in the future to support more
// AIE-specific vectorization patterns.

class AMDAIEVectorizationPass
    : public impl::AMDAIEVectorizationBase<AMDAIEVectorizationPass> {
 public:
  using AMDAIEVectorizationBase::AMDAIEVectorizationBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert</* tensor::TensorDialect, */ linalg::LinalgDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override;

  static bool hasOperandWithSmallElementType(Operation *op) {
    for (auto operand : op->getOperands()) {
      if (auto type = operand.getType().dyn_cast<ShapedType>()) {
        auto elementType = type.getElementType();
        if (elementType.getIntOrFloatBitWidth() <= 16) {
          return true;
        }
      }
    }
    return false;
  }
};

// A pattern to up-cast the result of a vector.contract as appropriate. This is
// designed to target AIE matmul instructions. For example AIE can do matmul
// with bf16 operands and f32 result, but not with bf16 operands and bf16
// result. So this pattern will up-cast the result to f32 if the result is bf16
// in such cases, and down-cast after the contract as appropriate.
//
// The position to perform this casting is up for debate: Could be really early
// in the pipeline, or could be pushed down to mlir-aie. Pros and cons to both.
struct UpcastVectorContractResult
    : public OpRewritePattern<mlir::vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    auto lhs = contractOp.getLhs();
    auto rhs = contractOp.getRhs();
    auto acc = contractOp.getAcc();

    auto lhsType = dyn_cast<VectorType>(lhs.getType());
    auto rhsType = dyn_cast<VectorType>(rhs.getType());
    auto accType = dyn_cast<VectorType>(acc.getType());

    if (!lhsType)
      return rewriter.notifyMatchFailure(contractOp,
                                         "lhs is not a vector type");
    if (!rhsType)
      return rewriter.notifyMatchFailure(contractOp,
                                         "rhs is not a vector type");
    if (!accType)
      return rewriter.notifyMatchFailure(contractOp,
                                         "acc is not a vector type");

    auto lhsElType = lhsType.getElementType();
    auto rhsElType = rhsType.getElementType();
    auto accElType = accType.getElementType();

    // For now we match the case where all lhs, rhs, acc are bf16.
    if (!lhsElType.isBF16() || !rhsElType.isBF16() || !accElType.isBF16())
      return rewriter.notifyMatchFailure(
          contractOp,
          "lhs, rhs, acc are not all bf16 (this constraint needs relaxing to "
          "support this patter for int types)");

    // Upcast acc to f32:
    auto f32Type = rewriter.getF32Type();
    auto f32VecType = VectorType::get(accType.getShape(), f32Type);
    auto newAcc = rewriter.create<arith::ExtFOp>(acc.getLoc(), f32VecType, acc);

    auto newContract = rewriter.create<vector::ContractionOp>(
        contractOp.getLoc(), lhs, rhs, newAcc, contractOp.getIndexingMaps(),
        contractOp.getIteratorTypes(), contractOp.getKind());

    // Downcast the new result to bf16 
    auto newResult = rewriter.create<arith::TruncFOp>(
        contractOp.getLoc(), accType, newContract.getResult());

    auto oldResult = contractOp.getResult();
    oldResult.replaceAllUsesWith(newResult);
    return success();
  }
};

void AMDAIEVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  IRRewriter rewriter(context);

  // Collect all operations which must be vectorized.
  SmallVector<Operation *> candidates;
  funcOp.walk([&](Operation *op) {
    // Only vectorize linalg ops (for now)
    if (!isa<linalg::LinalgOp>(op)) return;

    // iree-amd-aie's current tiling pipelines use linalg.copy ops to move data
    // between memory spaces. These copy ops should not be vectorized to
    // vector.transfer_read/transfer_write ops.
    if (isa<linalg::CopyOp>(op)) return;

    // Temporarily disabling linalg::FillOp vectorization. Current compilation
    // pipeline crashes in DMAToChannelPass: 'error: operand #0 does not
    // dominate this use'. TODO(newling) follow-up on this.
    if (isa<linalg::FillOp>(op)) return;

    // AIE architecture has no vector instructions for 32/64-bit types.
    if (!hasOperandWithSmallElementType(op)) return;

    candidates.push_back(op);
  });

  for (Operation *op : candidates) {
    (void)linalg::vectorize(rewriter, op);
  }

  RewritePatternSet vectorizationPatterns(context);

  vector::populateVectorReductionToContractPatterns(vectorizationPatterns);

  // Including this pattern prevents broadcasting in vector.transfer_read ops
  vector::populateVectorTransferPermutationMapLoweringPatterns(
      vectorizationPatterns);

  (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));

  RewritePatternSet set2(context);
  set2.add<UpcastVectorContractResult>(context);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(set2));
}
}  // namespace

std::unique_ptr<Pass> createAMDAIEVectorizationPass() {
  return std::make_unique<AMDAIEVectorizationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
