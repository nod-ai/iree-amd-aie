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

/// Utility function to check if any of the reduction dimension is being tiled.
static bool isTilingReductionDimension(TilingInterface consumerOp,
                                       SmallVector<int64_t> tileSizesVal) {
  SmallVector<utils::IteratorType> loopIteratorTypes =
      consumerOp.getLoopIteratorTypes();
  unsigned totalTileSizes = tileSizesVal.size();
  unsigned totalLoopIteratorTypes = loopIteratorTypes.size();
  // Assume the following cases for [parallel, parallel, reduction, parallel]
  // iter types :-
  // Case 1:
  //    tile_size = [8, 8] - this is essentially -> [8, 8, 0, 0], i.e. we tile
  //    both the parallel iter types and do not tile the last two iter type
  //    (reduction and parallel in this case).
  // Case 2:
  //    tile_size = [8, 0, 8] - this is essentially -> [8, 0, 8, 0], i.e. here
  //    we tile the first iter type (parallel), do not tile the second iter type
  //    (parallel), tile the third iter type (reduction) and do not tile the
  //    last iter type (parallel).
  // Case 3:
  //    tile_size = [0, 0, 8, 8] - here we only tile the last two iter types
  //    (reduction and parallel).
  if (totalTileSizes < totalLoopIteratorTypes) {
    tileSizesVal.append(totalLoopIteratorTypes - totalTileSizes, 0);
  }
  for (auto [tileSize, loopIteratorType] :
       llvm::zip(tileSizesVal, loopIteratorTypes)) {
    if (loopIteratorType == utils::IteratorType::reduction && tileSize != 0)
      return true;
  }
  return false;
}

static bool consumerToSkip(TilingInterface op) {
  if (isa<linalg::CopyOp>(op) || isa<tensor::UnPackOp>(op)) return true;
  return false;
}

LogicalResult applyTileAndFuse(RewriterBase &rewriter, TilingInterface rootOp,
                               DominanceInfo &dominanceInfo,
                               scf::SCFTileAndFuseOptions &tileAndFuseOptions) {
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
    // Find the next consumer op if it does not have loops OR it is from
    // the skip ops list which currently contains linalg.copy and tensor.unpack.
    if (op.getLoopIteratorTypes().empty() || consumerToSkip(op))
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
  } else if (isa<linalg::GenericOp>(consumerOp)) {
    // TODO(vivian): Fix in upstream. For now this is a temp hack to get around
    // the lowering config not passing to linalg.generic during lowering.
    tileSizesVal = {1, 1};
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
  if (!useSCFFor) {
    options.setMapping(
        {gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimY),
         gpu::GPUBlockMappingAttr::get(context, gpu::MappingId::DimX)});
  }

  if (!useSCFFor) {
    options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  }

  IRRewriter rewriter(context);
  DominanceInfo dominanceInfo(funcOp);
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(options);
  // We switch off fusion if any of the reduction dimension is being tiled. We
  // resort to the default fusion control function that eliminates certain ops
  // otherwise.
  if (bool tilingReductionDimension =
          isTilingReductionDimension(consumerOp, tileSizesVal)) {
    tileAndFuseOptions.setFusionControlFn(
        [&](tensor::ExtractSliceOp sliceOp, OpResult originalProducer,
            bool isDestinationOperand) -> std::tuple<bool, bool> {
          return {false, false};
        });
  } else {
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
  if (failed(applyTileAndFuse(rewriter, consumerOp, dominanceInfo,
                              tileAndFuseOptions))) {
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
