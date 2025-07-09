// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "iree-amdaie-tile"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// This pass tiles the TilingInterface operations. The `tilingLevel` must
/// be specified. It picks the `tilingLevel`-th list as tiling sizes from
/// lowering_config.
class AMDAIETilePass : public impl::AMDAIETileBase<AMDAIETilePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  AMDAIETilePass() = default;
  AMDAIETilePass(const AMDAIETilePass &pass) {}
  AMDAIETilePass(const AMDAIETileOptions &options) : AMDAIETileBase(options) {}

  void runOnOperation() override;
};

void AMDAIETilePass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  if (tilingLevel == -1) {
    LLVM_DEBUG(llvm::dbgs() << "tilingLevel not set, skip tiling\n");
    return;
  }

  // Currently, only tile linalg.copy which is the producer of lhs/rhs operand
  // of a contraction op.
  linalg::LinalgOp linalgOp;
  funcOp->walk([&](linalg::LinalgOp op) {
    if (linalg::isaContractionOpInterface(op)) {
      linalgOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!linalgOp) {
    LLVM_DEBUG(llvm::dbgs()
               << "----- skip, user linalg op is not found -----\n");
    return;
  }

  auto lhsOp = linalgOp->getOperand(0).getDefiningOp();
  auto rhsOp = linalgOp->getOperand(1).getDefiningOp();
  if (!isa_and_present<linalg::CopyOp>(lhsOp) ||
      !isa_and_present<linalg::CopyOp>(rhsOp)) {
    LLVM_DEBUG(llvm::dbgs()
               << "----- skip, producer is not linalg.copy -----\n");
    return;
  }

  // The lowering config is added to the original matmul/generic op. To tile the
  // linalg.copy op, get lowering configs from its user op.
  SmallVector<int64_t> tileSizesVal;
  if (auto loweringConfig =
          getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(linalgOp)) {
    tileSizesVal = loweringConfig.getTileSizeVals(tilingLevel);
  } else {
    FailureOr<IREE::Codegen::LoweringConfigAttr> maybeLoweringConfig =
        getFirstLoweringConfig<IREE::Codegen::LoweringConfigAttr>(
            getComputeOps(funcOp));
    if (failed(maybeLoweringConfig)) {
      LLVM_DEBUG(llvm::dbgs() << "can't find lowering_config, skip tiling");
      return;
    }
    tileSizesVal = maybeLoweringConfig.value().getTileSizeVals(tilingLevel);
  }

  // Input tile sizes should be [0, 0, tileK].
  if (tileSizesVal.size() != 3 || tileSizesVal[2] <= 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "----- skip, tile sizes are not correct -----\n");
    return;
  }
  const int64_t tileK = tileSizesVal[2];
  SmallVector<int64_t> lhsTileSizes = {0, tileK};
  // Currently matmul and transpose op are the only ones supported. If its not
  // matmul then it is a transpose.
  SmallVector<int64_t> rhsTileSizes;
  if (isa<linalg::MatmulOp>(linalgOp)) {
    rhsTileSizes = {tileK, 0};
  } else {
    rhsTileSizes = {0, tileK};
  }
  SmallVector<SmallVector<int64_t>> allTileSizes = {lhsTileSizes, rhsTileSizes};
  SmallVector<TilingInterface> tilingOps = {cast<TilingInterface>(lhsOp),
                                            cast<TilingInterface>(rhsOp)};

  IRRewriter rewriter(context);
  for (auto [op, sizes] : llvm::zip(tilingOps, allTileSizes)) {
    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(context, sizes);
    auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);

    FailureOr<scf::SCFTilingResult> tiledResults =
        scf::tileUsingSCF(rewriter, op, options);
    if (failed(tiledResults)) {
      op->emitOpError("failed to tile the operation");
      return signalPassFailure();
    }
    rewriter.replaceOp(op, tiledResults->replacements);
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIETilePass(AMDAIETileOptions options) {
  return std::make_unique<AMDAIETilePass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
