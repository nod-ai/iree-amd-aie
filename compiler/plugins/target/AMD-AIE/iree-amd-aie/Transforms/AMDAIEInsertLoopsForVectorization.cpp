// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

// Change linalg.generics which are matmul-like (parallel parallel reduction) to
// have the outer dimensions tiled to size 1, with scf.for loops over the outer
// dimensions.
class OuterDimsToLoopsForMatmulLikePattern
    : public OpRewritePattern<linalg::GenericOp> {
 public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    auto iteratorTypes = genericOp.getIteratorTypesArray();
    auto numIterators = iteratorTypes.size();

    if (numIterators < 4) {
      return rewriter.notifyMatchFailure(
          genericOp,
          llvm::formatv("The number of iterators ({0}) is less than 4, so "
                        "there are no outer dimensions to tile.",
                        numIterators));
    }

    if (genericOp->getNumOperands() != 3) {
      return rewriter.notifyMatchFailure(
          genericOp, llvm::formatv("The number of operands ({0}) is not 3, so "
                                   "this is not matmul-like.",
                                   genericOp->getNumOperands()));
    }

    // Don't transform to scf.for loops unless there is at least one
    // non-singleton loop to construct (this ensures termination of the pass).
    {
      std::array<int64_t, 3> nTrailingNonSingleton;
      for (auto operand : llvm::enumerate(genericOp->getOperands())) {
        nTrailingNonSingleton[operand.index()] =
            getNumDimsFromFirstNonSingleton(operand.value());
      }

      bool allLessThan3 =
          llvm::all_of(nTrailingNonSingleton, [](int64_t n) { return n < 3; });

      if (allLessThan3) {
        return rewriter.notifyMatchFailure(
            genericOp,
            llvm::formatv(
                "The number of non-singleton trailing dimensions in each "
                "operand is less than 3: {0}, {1}, and {2} respectively.",
                nTrailingNonSingleton[0], nTrailingNonSingleton[1],
                nTrailingNonSingleton[2]));
      }
    }

    assert(iteratorTypes.size() >= 3 && "expected at least 3 iterators here");

    auto hasMatmulIterators = [&]() {
      return linalg::isParallelIterator(
                 iteratorTypes[iteratorTypes.size() - 3]) &&
             linalg::isParallelIterator(
                 iteratorTypes[iteratorTypes.size() - 2]) &&
             linalg::isReductionIterator(
                 iteratorTypes[iteratorTypes.size() - 1]);
    }();

    if (!hasMatmulIterators) {
      return rewriter.notifyMatchFailure(
          genericOp,
          "The final 3 dimensions do not match matmul iterators (should be "
          "'parallel parallel reduction').");
    }

    // Check that the mapped dimensions correspond to matmul.
    // Specifically the final dimensions of the maps are
    // operand 0 : (i, j, k) -> (i, k)
    // operand 1 : (i, j, k) -> (k, j)
    // result    : (i, j, k) -> (i, j)

    auto indexingMaps = genericOp.getIndexingMaps();
    assert(indexingMaps.size() == 3 && "expected 3 indexing maps here");

    AffineMap map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
    AffineMap map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
    AffineMap map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

    auto nResults0 = map0.getNumResults();
    auto nResults1 = map1.getNumResults();
    auto nResults2 = map2.getNumResults();

    if (map0.getResult(nResults0 - 2) != map2.getResult(nResults2 - 2))
      return rewriter.notifyMatchFailure(
          genericOp,
          "unable to locate 'M' in affine maps [(M,K) x (K,N) -> (M,N)]");
    if (map1.getResult(nResults1 - 1) != map2.getResult(nResults2 - 1))
      return rewriter.notifyMatchFailure(
          genericOp,
          "unable to locate 'N' in affine maps [(M,K) x (K,N) -> (M,N)]");
    if (map0.getResult(nResults0 - 1) != map1.getResult(nResults1 - 2))
      return rewriter.notifyMatchFailure(
          genericOp,
          "unable to locate 'K' in affine maps [(M,K) x (K,N) -> (M,N)]");

    // Tile all dimensions except the 3 inner-most dimensions. Tile size '1'
    // denotes tiling with smallest possible tile (size 1) and tile size '0'
    // denotes tiling with the largest possible tile (size equal to the
    // dimension size). The tiles sizes are thus [1,1,...1,0,0,0].
    SmallVector<int64_t> tileSizes(numIterators, 1);
    tileSizes[numIterators - 3] = 0;
    tileSizes[numIterators - 2] = 0;
    tileSizes[numIterators - 1] = 0;
    auto outerOpts = linalg::LinalgTilingOptions().setTileSizes(tileSizes);
    auto outerTiled = linalg::tileLinalgOp(rewriter, genericOp, outerOpts);
    rewriter.replaceOp(genericOp, outerTiled.value().loops[0]->getResult(0));
    return success();
  }

 private:
  // Example: (1,1,4,1,2,3,4) has 5 dimensions from the first non-singleton.
  static int getNumDimsFromFirstNonSingleton(Value v) {
    auto shape = v.getType().cast<ShapedType>().getShape();
    auto nDims = shape.size();
    auto firstNonSingleton = [&]() -> int {
      for (auto i : llvm::enumerate(shape)) {
        if (i.value() > 1) {
          return i.index();
        }
      }
      return nDims;
    }();
    return nDims - firstNonSingleton;
  };
};

class AMDAIEInsertLoopsForVectorizationPass
    : public impl::AMDAIEInsertLoopsForVectorizationBase<
          AMDAIEInsertLoopsForVectorizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect>();
  }
  void runOnOperation() override;
};

void AMDAIEInsertLoopsForVectorizationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto operation = getOperation();
  IRRewriter rewriter(context);
  RewritePatternSet patterns(operation.getContext());
  patterns.add<OuterDimsToLoopsForMatmulLikePattern>(context);
  auto converged = applyPatternsAndFoldGreedily(operation, std::move(patterns));
  (void)converged;
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertLoopsForVectorizationPass() {
  return std::make_unique<AMDAIEInsertLoopsForVectorizationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
