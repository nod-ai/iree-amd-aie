// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"


namespace mlir::iree_compiler::AMDAIE {

namespace {

/*class TileUsingSCFForallOp : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {

  MLIRContext *context = matmulOp->getContext();
  SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(context, {8,8});
  auto tilingOptions =
      scf::SCFTilingOptions().setTileSizes(tileSizes);
  FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCFForallOp(rewriter, cast<TilingInterface>(matmulOp.getOperation()), tilingOptions);

    return success();
  }
};*/

class AMDAIETileToSCFForAllPass
    : public AMDAIETileToSCFForAllBase<AMDAIETileToSCFForAllPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  AMDAIETileToSCFForAllPass() = default;
  AMDAIETileToSCFForAllPass(const AMDAIETileToSCFForAllPass &pass){};

  void runOnOperation() override;
};

void AMDAIETileToSCFForAllPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
 /*patterns.add<TileUsingSCFForallOp>(context);
  if (failed(
          applyPatterns(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
*/
  }

}  // namespace

std::unique_ptr<OperationPass<>> createAMDAIETileToSCFForAllPass() {
  return std::make_unique<AMDAIETileToSCFForAllPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
