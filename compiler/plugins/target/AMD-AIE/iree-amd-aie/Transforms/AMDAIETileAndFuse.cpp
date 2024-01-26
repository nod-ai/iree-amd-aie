// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-tile-and-fuse"

namespace mlir::iree_compiler::AMDAIE {

namespace {

LogicalResult applyTileAndFuse(RewriterBase &rewriter, TilingInterface rootOp,
                               DominanceInfo &dominanceInfo,
                               scf::SCFTilingOptions options, bool useSCFFor,
                               bool useFusion) {
  if (!useSCFFor) {
    options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  }
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(options);
  tileAndFuseOptions.setFusionControlFn(
      [&](tensor::ExtractSliceOp sliceOp, OpResult originalProducer,
          bool isDestinationOperand) -> std::tuple<bool, bool> {
        bool fusableOp =
            TypeSwitch<Operation *, bool>(originalProducer.getOwner())
                // List ops that shouldnt be fused.
                .Case<tensor::PackOp, tensor::PadOp, linalg::CopyOp,
                      memref::CopyOp>([](Operation *) { return false; })
                // Fuse all Linalg ops (can be generalized later)
                .Default([&](Operation *op) {
                  return op->getDialect() ==
                         rewriter.getContext()
                             ->getLoadedDialect<linalg::LinalgDialect>();
                });
        return {fusableOp, false};
      });

  if (useFusion) {
    tileAndFuseOptions.setFusionControlFn(
        [&](tensor::ExtractSliceOp sliceOp, OpResult originalProducer,
            bool isDestinationOperand) -> std::tuple<bool, bool> {
          bool fusableOp =
              TypeSwitch<Operation *, bool>(originalProducer.getOwner())
                  // List ops that shouldnt be fused.
                  .Case<tensor::PackOp, tensor::PadOp, linalg::CopyOp,
                        memref::CopyOp>([](Operation *) { return false; })
                  // Fuse all Linalg ops (can be generalized later)
                  .Default([&](Operation *op) {
                    return op->getDialect() ==
                           rewriter.getContext()
                               ->getLoadedDialect<linalg::LinalgDialect>();
                  });
          return {fusableOp, false};
        });
  }
  // If user of pass requests they dont want fusion we disable all fusion.
  else {
    tileAndFuseOptions.setFusionControlFn(
        [&](tensor::ExtractSliceOp sliceOp, OpResult originalProducer,
            bool isDestinationOperand) -> std::tuple<bool, bool> {
          return {false, false};
        });
  }

  FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, rootOp,
                                                tileAndFuseOptions);
  if (failed(tileAndFuseResult)) {
    return rootOp.emitOpError("failed to tile and fuse with op as root");
  }

  for (auto [origVal, replacement] : tileAndFuseResult->replacements) {
    rewriter.replaceUsesWithIf(origVal, replacement, [&](OpOperand &use) {
      return !isa<tensor::DimOp>(use.getOwner());
    });
  }

  return success();
}

/// This pass starts with the last TilingInterface operation, tiles the op and
/// fuses its producers recursively. The `tilingLevel` must be specified. It
/// picks the `tilingLevel`-th list as tiling sizes from lowering_config.
class AMDAIETileAndFusePass
    : public impl::AMDAIETileAndFuseBase<AMDAIETileAndFusePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, scf::SCFDialect>();
  }

  AMDAIETileAndFusePass() = default;
  AMDAIETileAndFusePass(const AMDAIETileAndFusePass &pass) {}
  AMDAIETileAndFusePass(const AMDAIETileAndFuseOptions &options)
      : AMDAIETileAndFuseBase(options) {}

  void runOnOperation() override;
};

void AMDAIETileAndFusePass::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp funcOp = getOperation();

  TilingInterface consumerOp;
  funcOp->walk<WalkOrder::PostOrder, ReverseIterator>([&](TilingInterface op) {
    // Find the next consumer op if it does not have loops OR if it is a
    // linalg.copy op.
    if (op.getLoopIteratorTypes().empty() || isa<linalg::CopyOp>(op))
      return WalkResult::advance();
    consumerOp = op;
    return WalkResult::interrupt();
  });
  if (!consumerOp) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, no consumer op -----\n");
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "consumerOp: " << consumerOp << "\n");
  LLVM_DEBUG(llvm::dbgs() << "tilingLevel: " << tilingLevel << "\n");
  // TODO(avarma): Have a global CONSTANT defining tiling stages and the
  // tiling strategy.
  // If `consumerOp` has its own lowering config, we prefer using it.
  // Otherwise, fallback to find a lowering_config from other operations.
  SmallVector<int64_t> tileSizesVal;
  if (auto loweringConfig = getLoweringConfig(consumerOp)) {
    tileSizesVal = loweringConfig.getTileSizeVals(tilingLevel);
  } else {
    FailureOr<IREE::Codegen::LoweringConfigAttr> maybeLoweringConfig =
        getLoweringConfig(getComputeOps(funcOp));
    if (failed(maybeLoweringConfig)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "can't find lowering_config, skip TileAndFuse");
      return;
    }
    tileSizesVal = maybeLoweringConfig.value().getTileSizeVals(tilingLevel);
  }

  if (llvm::all_of(tileSizesVal, [&](int64_t size) { return size == 0; })) {
    LLVM_DEBUG(llvm::dbgs() << "----- skip, all zeros -----\n");
    return;
  }

  SmallVector<OpFoldResult> tileSizes =
      getAsIndexOpFoldResult(context, tileSizesVal);
  auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);
  // When tiling using scf.for we do not need to set any mapping.
  if (tilingLevel != 2) {
    options.setMapping(
        {gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimY),
         gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimX)});
  }

  IRRewriter rewriter(context);
  DominanceInfo dominanceInfo(funcOp);
  if (failed(applyTileAndFuse(rewriter, consumerOp, dominanceInfo, options,
                              useSCFFor, useFusion))) {
    LLVM_DEBUG(llvm::dbgs() << "----- tile and fuse failed -----\n");
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIETileAndFusePass(
    AMDAIETileAndFuseOptions options) {
  return std::make_unique<AMDAIETileAndFusePass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
