// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEInsertLoopsForVectorizationPass
    : public impl::AMDAIEInsertLoopsForVectorizationBase<
          AMDAIEInsertLoopsForVectorizationPass> {
 private:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect>();
  }

  // The number of dimensions ignoring leading 0s and 1s.
  // Examples:
  //   (5,5,5) -> 3
  //   (1,5,5) -> 2
  //   (1,1,5) -> 1
  //   (1,1,1) -> 0
  static int getNumInnerDims(Value v) {
    auto shape = cast<ShapedType>(v.getType()).getShape();
    auto nDims = shape.size();

    // The first dimension which is not 0 or 1.
    auto firstDim = [&]() -> int {
      for (auto i : llvm::enumerate(shape)) {
        if (i.value() > 1) {
          return i.index();
        }
      }
      return nDims;
    }();
    return nDims - firstDim;
  };

  /// Tile the generic op using `tileSizes` and coalesce the generated tiling
  /// loops in order to minimize the overhead of loop control/branch statements.
  /// This function can work on both tensor as well as memref inputs.
  static void performTiling(IRRewriter &rewriter, linalg::GenericOp genericOp,
                            SmallVector<int64_t> &tileSizes) {
    auto opts = linalg::LinalgTilingOptions().setTileSizes(tileSizes);
    auto tiled = linalg::tileLinalgOp(rewriter, genericOp, opts);
    const auto &tileLoops = tiled.value().loops;
    SmallVector<scf::ForOp> loops = llvm::map_to_vector(
        tileLoops, [](Operation *loop) { return cast<scf::ForOp>(loop); });
    if (genericOp->getResults().size()) {
      rewriter.replaceOp(genericOp, loops[0]->getResult(0));
    } else {
      rewriter.eraseOp(genericOp);
    }
    return;
  }
  // Tile all dimensions except the 3 inner-most dimensions. Tile size '1'
  // denotes tiling with smallest possible tile (size 1) and tile size '0'
  // denotes tiling with the largest possible tile (size equal to the
  // dimension size). The tile sizes we use are [1,1,...1,0,0,0].
  static std::optional<SmallVector<int64_t>> formTileSizesForMatmul(
      linalg::GenericOp genericOp) {
    auto iteratorTypes = genericOp.getIteratorTypesArray();
    auto numIterators = iteratorTypes.size();
    assert(numIterators >= 3 && "expected at least 3 iterators here");

    SmallVector<int64_t> tileSizes(numIterators, 1);
    tileSizes[numIterators - 3] = 0;
    tileSizes[numIterators - 2] = 0;
    tileSizes[numIterators - 1] = 0;
    return tileSizes;
  }

  /// We tile all but the innermost two dimensions currently because they form
  /// the smallest tiled M x N dimension of the matmul.
  static std::optional<SmallVector<int64_t>> formTileSizesForElementwise(
      linalg::GenericOp genericOp) {
    auto iteratorTypes = genericOp.getIteratorTypesArray();
    auto numIterators = iteratorTypes.size();
    assert(numIterators >= 2 && "expected at least 2 iterators here");
    SmallVector<int64_t> tileSizes(numIterators, 1);
    tileSizes[numIterators - 2] = 0;
    tileSizes[numIterators - 1] = 0;
    return tileSizes;
  }

  /// Collapse unit dims of the generic op before tiling for vectorization. Since
  /// this is optinal we need not return failure if the collapsing cannot take
  /// place. Eg: For <2x3x4> since there aren't any unit dimensions, it'd return
  /// failure, hence we can simply return.
  static void collapseUnitDims(IRRewriter &rewriter,
                               linalg::GenericOp &genericOp) {
    linalg::ControlDropUnitDims options;
    options.rankReductionStrategy =
        linalg::ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice;
    FailureOr<linalg::DropUnitDimsResult> result =
        linalg::dropUnitDims(rewriter, genericOp, options);
    if (failed(result)) return;
    if (genericOp->getResults().size()) {
      rewriter.replaceOp(genericOp, result->replacements);
    } else {
      rewriter.eraseOp(genericOp);
    }
    genericOp = result->resultOp;
    return;
  }

  // Return success if the generic op is rewritten, failure otherwise.
  LogicalResult maybeRewrite(linalg::GenericOp genericOp,
                             IRRewriter &rewriter) {
    if (isa<linalg::CopyOp, linalg::FillOp>(genericOp)) return failure();

    auto iteratorTypes = genericOp.getIteratorTypesArray();
    auto numIterators = iteratorTypes.size();

    // No outer dimensions to tile if fewer than 3 iterators.
    if (numIterators < 3) return failure();

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(genericOp);
    // Enable generating loops for vectorization in case of element-wise ops.
    if (llvm::all_of(iteratorTypes, [&](mlir::utils::IteratorType iterator) {
          return linalg::isParallelIterator(iterator);
        })) {
      collapseUnitDims(rewriter, genericOp);
      std::optional<SmallVector<int64_t>> tileSizes =
          formTileSizesForElementwise(genericOp);
      if (!tileSizes) {
        return genericOp->emitOpError()<<"unable to form tile sizes for the elementwise op";
      }
      performTiling(rewriter, genericOp, *tileSizes);
      return success();
    }
    // Matmul-like ops have 3 operands.
    if (genericOp->getNumOperands() != 3) return failure();

    // Check that the operands and result are of vectorizable types, if they are
    // not, then do not tile.
    auto hasAieVectorizableTypes = [genericOp]() -> bool {
      auto elType = [](Value v) {
        return cast<ShapedType>(v.getType()).getElementType();
      };
      auto lhsType = elType(genericOp->getOperand(0));
      auto rhsType = elType(genericOp->getOperand(1));
      auto resType = elType(genericOp->getOperand(2));
      auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(
          genericOp->getParentOfType<func::FuncOp>());
      std::optional<AMDAIE::AMDAIEDevice> maybeDevice =
          mlir::iree_compiler::AMDAIE::getConfigAMDAIEDevice(targetAttr);
      if (!maybeDevice) {
        genericOp->emitOpError() << "has no AMDAIEDevice in the target "
                                    "attribute configuration. This "
                                    "device-specific information is required "
                                    "to determine what vector "
                                    "sizes are supported.";
        return false;
      }
      AMDAIE::AMDAIEDeviceModel deviceModel =
          AMDAIE::getDeviceModel(maybeDevice.value());
      FailureOr<std::array<uint32_t, 3>> maybeSize =
          deviceModel.getAIEMatmulInstructionSize(lhsType, rhsType, resType);
      return !failed(maybeSize);
    }();
    if (!hasAieVectorizableTypes) return failure();

    // Don't transform to scf.for loops unless there is at least one
    // non-singleton loop to construct. This isn't strictly necessary, but
    // avoids generating a bunch of loops of size 1.
    if (llvm::all_of(genericOp->getOperands(), [&](Value operand) {
          return getNumInnerDims(operand) < 3;
        }))
      return failure();

    assert(iteratorTypes.size() >= 3 && "expected at least 3 iterators here");

    // Check that innermost 3 iterators are 'parallel, parallel, reduction'.
    for (auto i : {2, 3}) {
      if (!linalg::isParallelIterator(iteratorTypes[numIterators - i]))
        return failure();
    }
    if (!linalg::isReductionIterator(iteratorTypes[iteratorTypes.size() - 1]))
      return failure();

    // Check that the 'parallel, parallel, reduction' map exactly to a matmul,
    // or a matmul_transpose_b.
    {
      auto indexingMaps = genericOp.getIndexingMaps();
      assert(indexingMaps.size() == 3 && "expected 3 indexing maps here");
      auto getDim = [&](uint32_t mapIndex, uint32_t matMulIndex) {
        auto aMap = cast<AffineMapAttr>(indexingMaps[mapIndex]).getValue();
        auto nResults = aMap.getNumResults();
        return aMap.getResult(nResults - 2 + matMulIndex);
      };
      uint32_t A = 0, B = 1, C = 2;

      auto isMatmul = getDim(A, 0) == getDim(C, 0) &&  // M
                      getDim(B, 1) == getDim(C, 1) &&  // N
                      getDim(A, 1) == getDim(B, 0);    // K

      auto isMatmulTransposeB = getDim(A, 0) == getDim(C, 0) &&  // M
                                getDim(B, 0) == getDim(C, 1) &&  // N
                                getDim(A, 1) == getDim(B, 1);    // K

      if (!isMatmul && !isMatmulTransposeB) return failure();
    }

    collapseUnitDims(rewriter, genericOp);
    std::optional<SmallVector<int64_t>> tileSizes =
        formTileSizesForMatmul(genericOp);
    if (!tileSizes) {
      return genericOp->emitOpError()<<"unable to form tile sizes for the matmul op";
    }
    performTiling(rewriter, genericOp, *tileSizes);
    return success();
  }

  void runOnOperation() final {
    MLIRContext *context = &getContext();
    mlir::FunctionOpInterface operation = getOperation();

    IRRewriter rewriter(context);
    operation->walk([&](linalg::GenericOp genericOp) {
      (void)maybeRewrite(genericOp, rewriter);
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertLoopsForVectorizationPass() {
  return std::make_unique<AMDAIEInsertLoopsForVectorizationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
