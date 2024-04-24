// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/KernelDispatch.h"

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "kernel-dispatch"

namespace mlir::iree_compiler::AMDAIE {

using detail::findLargestFactor;

static SmallVector<int64_t> getPackedSize(linalg::LinalgOp linalgOp,
                                          const int packLevel, int m = 0,
                                          int n = 0, int k = 0) {
  // TODO (newling): consider emiting an error/warning if the default sizes are used as a
  // fallback.
  SmallVector<int64_t> defaultSizes;
  // TODO (nmeshram) : We should not need this and be able to fix the pack
  // config after we have padding support
  int minM = m ? findLargestFactor(m, 4) : 4;
  int minN = n ? findLargestFactor(n, 4) : 4;
  int minK = k ? findLargestFactor(k, 8) : 8;
  if (packLevel == 1) {
    defaultSizes = {{minM, minN, minK}};
  } else if (packLevel == 2) {
    defaultSizes = {{0, 0, 0, minM, minN, minK}};
  } else {
    linalgOp->emitError("invalid value of pack level.");
  }
  if (!isa<linalg::MatmulOp>(linalgOp)) {
    return defaultSizes;
  }

  auto getElementType = [](Value v) {
    return v.getType().cast<ShapedType>().getElementType();
  };

  auto elTypeLhs = getElementType(linalgOp->getOperand(0));
  auto elTypeRhs = getElementType(linalgOp->getOperand(1));
  auto elTypeAcc = getElementType(linalgOp->getResult(0));

  auto maybeInstructionSize =
      getAIEMatmulInstructionSize(elTypeLhs, elTypeRhs, elTypeAcc);

  if (failed(maybeInstructionSize)) {
    return defaultSizes;
  }

  auto instructionSize = maybeInstructionSize.value();
  SmallVector<int64_t> packedSizes(3, 0);
  std::copy(instructionSize.begin(), instructionSize.end(),
            packedSizes.begin());
  if (packLevel == 2) {
    packedSizes.insert(packedSizes.begin(), {0, 0, 0});
  }
  return packedSizes;
}

static LogicalResult setRootConfigForPackPeelPipeline(
    mlir::FunctionOpInterface entryPointFn, linalg::LinalgOp linalgOp,
    AIEConfig cfg) {
  auto initType =
      llvm::cast<ShapedType>(linalgOp.getDpsInitOperand(0)->get().getType());
  auto initShape = initType.getShape();
  auto lhsType =
      llvm::cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto lhsShape = lhsType.getShape();

  FailureOr<unsigned> maybeTilingScaleFactor =
      getTilingScaleFactor(initType.getElementType());
  if (failed(maybeTilingScaleFactor)) {
    return linalgOp.emitOpError("expected bitwidth 64/32/16/8");
  }
  unsigned tilingScaleFactor = maybeTilingScaleFactor.value();
  auto tileM0 = findLargestFactor((int)initShape[0], 32 * tilingScaleFactor);
  auto tileN0 = findLargestFactor((int)initShape[1], 32 * tilingScaleFactor);

  // ------------------------------------------------------
  // --------------- Set packing config -------------------
  // ------------------------------------------------------
  MLIRContext *context = entryPointFn.getContext();
  // Pack level => 1.
  auto packedK0 = findLargestFactor((int)lhsShape[1], 16 * tilingScaleFactor);
  SmallVector<int64_t> packedSizes = {tileM0, tileN0, packedK0};
  // Transpose B matrix from [K N n k] to [K N k n]
  SmallVector<int64_t> transposePackIndices = {1};
  // There is no corresponding unpack for the specified pack operation
  // 0 is used when unpack is empty
  SmallVector<bool> unpackEmpty = {false};
  SmallVector<SmallVector<int64_t>> innerPerm = {{1, 0}};
  SmallVector<SmallVector<int64_t>> outerPerm = {{0, 1}};
  auto packingConfigLevel1Attr = getPackingConfigPackingLevelAttr(
      context, packedSizes, transposePackIndices, unpackEmpty, innerPerm,
      outerPerm);

  // Pack level => 2.
  // packed size for [M, N, K, m, n, k]
  const int packLevel = 2;
  packedSizes = getPackedSize(linalgOp, packLevel);
  // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
  // Transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
  // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
  transposePackIndices = {0, 1, 2};
  // Only the third pack operation has a corresponding unpack operation
  unpackEmpty = {false, false, true};
  innerPerm = {{0, 1}, {1, 0}, {0, 1}};
  outerPerm = {{0, 1, 3, 2}, {0, 1, 3, 2}, {0, 1, 3, 2}};
  auto packingConfigLevel2Attr = getPackingConfigPackingLevelAttr(
      context, packedSizes, transposePackIndices, unpackEmpty, innerPerm,
      outerPerm);

  SmallVector<PackingConfigPackingLevelAttr> packingConfigLevelsVal = {
      packingConfigLevel1Attr, packingConfigLevel2Attr};
  auto packingConfigLevels =
      PackingConfigPackingLevelsAttr::get(context, packingConfigLevelsVal);
  auto config = PackingConfigAttr::get(context, packingConfigLevels);
  setPackingConfig(linalgOp, config);

  // ------------------------------------------------------
  // -------------- Set lowering config -------------------
  // ------------------------------------------------------
  // Currently, assume working on a 2x2 AIE array, so the second level tile
  // sizes should be (tileM0/2, tileN0/2). Considering the packing sizes, the
  // adjusted tile sizes should be (tileM0/2/packedM1, tileN0/2/packedN1).
  auto packedM1 = packedSizes[3];
  auto packedN1 = packedSizes[4];
  auto tileM1 = findLargestFactor((int)tileM0 / packedM1,
                                  16 * tilingScaleFactor / packedM1);
  auto tileN1 = findLargestFactor((int)tileN0 / packedN1,
                                  16 * tilingScaleFactor / packedN1);
  // Set tile size for K as constant 1, so that the packed outer K dimension
  // is 1.
  const int tileK = 1;

  SmallVector<int64_t> TileSizeLevel0 = {tileM0, tileN0};
  SmallVector<int64_t> TileSizeLevel1 = {0, 0, tileK};
  SmallVector<int64_t> TileSizeLevel2 = {0, 0, 0, tileM1, tileN1, 0};
  TileSizesListType tileSizes = {TileSizeLevel0, TileSizeLevel1,
                                 TileSizeLevel2};
  if (failed(setOpConfigAndEntryPointFnTranslation(
          entryPointFn, linalgOp, tileSizes,
          IREE::Codegen::DispatchLoweringPassPipeline::Custom))) {
    return failure();
  }
  return success();
}

static LogicalResult setRootConfigForPadPackPipeline(
    mlir::FunctionOpInterface entryPointFn, linalg::LinalgOp linalgOp,
    AIEConfig cfg) {
  // Assume working on a 4x4 AIE array. Currently, the tile sizes are chosen
  // empirically for large GEMM sizes, which are [64*s, 64*s, 256] for the first
  // level and [16*s, 16*s, 16*s] for the second level, where 's' is the scaling
  // scaling factor based on the element type's bit width. Basic min/max
  // constraints are added to avoid failure for small GEMM sizes.
  auto initType = linalgOp.getDpsInitOperand(0)->get().getType();
  auto initShape = llvm::cast<ShapedType>(initType).getShape();
  auto lhsType =
      llvm::cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto lhsShape = lhsType.getShape();

  FailureOr<unsigned> maybeTilingScaleFactor =
      getTilingScaleFactor(lhsType.getElementType());
  if (failed(maybeTilingScaleFactor)) {
    return linalgOp.emitOpError("expected bitwidth 64/32/16/8");
  }
  unsigned tilingScaleFactor = maybeTilingScaleFactor.value();

  // Do packing first to allow better packing configs
  // ------------------------------------------------------
  // --------------- Set packing config -------------------
  // ------------------------------------------------------
  MLIRContext *context = entryPointFn.getContext();
  const int packLevel = 1;
  auto packedSizes = getPackedSize(linalgOp, packLevel, (int)initShape[0],
                                   (int)initShape[1], (int)lhsShape[1]);
  SmallVector<int64_t> transposePackIndices = {0, 1, 2};
  SmallVector<bool> unpackEmpty = {false, false, true};
  SmallVector<SmallVector<int64_t>> innerPerm = {{0, 1}, {1, 0}, {0, 1}};
  SmallVector<SmallVector<int64_t>> outerPerm = {{1, 0}, {1, 0}, {1, 0}};
  auto packingConfigLevel1Attr = getPackingConfigPackingLevelAttr(
      context, packedSizes, transposePackIndices, unpackEmpty, innerPerm,
      outerPerm);
  SmallVector<PackingConfigPackingLevelAttr> packingConfigLevelsVal = {
      packingConfigLevel1Attr};

  auto packingConfigLevels =
      PackingConfigPackingLevelsAttr::get(context, packingConfigLevelsVal);
  auto config = PackingConfigAttr::get(context, packingConfigLevels);
  setPackingConfig(linalgOp, config);

  // Do tiling
  // ------------------------------------------------------
  // -------------- Set lowering config -------------------
  // ------------------------------------------------------
  // TODO (nmeshram) : We should be able to use fixed tiling config after we
  // have padding support.
  auto tileM1 = findLargestFactor((int)initShape[0], 16 * tilingScaleFactor,
                                  (int)packedSizes[0]);
  auto tileN1 = findLargestFactor((int)initShape[1], 16 * tilingScaleFactor,
                                  (int)packedSizes[1]);
  auto tileK1 = findLargestFactor((int)lhsShape[1] / (int)packedSizes[2],
                                  2 * tilingScaleFactor);

  auto tileM0 =
      findLargestFactor((int)initShape[0], 64 * tilingScaleFactor, (int)tileM1);
  auto tileN0 =
      findLargestFactor((int)initShape[1], 64 * tilingScaleFactor, (int)tileN1);
  auto tileK0 = findLargestFactor((int)lhsShape[1], 256,
                                  (int)tileK1 * (int)packedSizes[2]);

  SmallVector<int64_t> TileSizeLevel0 = {tileM0, tileN0};
  SmallVector<int64_t> TileSizeLevel1 = {0, 0, tileK0};
  SmallVector<int64_t> TileSizeLevel2 = {tileM1, tileN1};
  SmallVector<int64_t> TileSizeLevel3 = {0, 0, tileK1};
  TileSizesListType tileSizes = {TileSizeLevel0, TileSizeLevel1, TileSizeLevel2,
                                 TileSizeLevel3};
  if (failed(setOpConfigAndEntryPointFnTranslation(
          entryPointFn, linalgOp, tileSizes,
          IREE::Codegen::DispatchLoweringPassPipeline::Custom))) {
    return failure();
  }

  return success();
}

/// TODO(avarma): This currently is skipping checking for ext* ops.
static bool bodyMatcherForMatmulTranspose(Value yieldVal, Block *body) {
  Operation *addOp = yieldVal.getDefiningOp();
  if (!isa_and_nonnull<arith::AddIOp, arith::AddFOp>(addOp)) {
    return false;
  }
  Operation *mulOp = addOp->getOperand(1).getDefiningOp();
  if (!isa_and_nonnull<arith::MulIOp, arith::MulFOp>(mulOp)) {
    return false;
  }
  auto lhsBlockArg = mulOp->getOperand(0).dyn_cast<BlockArgument>();
  auto rhsBlockArg = mulOp->getOperand(1).dyn_cast<BlockArgument>();
  auto outBlockArg = addOp->getOperand(0).dyn_cast<BlockArgument>();
  if (!lhsBlockArg || !rhsBlockArg || !outBlockArg ||
      lhsBlockArg.getOwner() != body || rhsBlockArg.getOwner() != body ||
      outBlockArg.getOwner() != body || lhsBlockArg.getArgNumber() != 0 ||
      rhsBlockArg.getArgNumber() != 1 || outBlockArg.getArgNumber() != 2) {
    return false;
  }
  return true;
}

/// `isMatmulTranspose` is a utility function that aims to indentify whether a
/// linalg.generic op is a matmul transpose op.
static bool isMatmulTranspose(linalg::GenericOp genericOp) {
  // Step 1. Test the body of the generic to indeed be what we expect for a
  //         matmul transpose.
  Block *body = genericOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  if (!bodyMatcherForMatmulTranspose(yieldVal, body)) {
    return false;
  }
  // Step 2. Check iterator types.
  SmallVector<utils::IteratorType> matmulTransposeIteratorTypes = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};
  SmallVector<utils::IteratorType> opIteratorTypes =
      genericOp.getIteratorTypesArray();
  if (matmulTransposeIteratorTypes != opIteratorTypes) {
    return false;
  }
  // Step 3. Test the indexing maps.
  ArrayAttr indexingMaps = genericOp.getIndexingMaps();
  if (indexingMaps.size() != 3) return false;

  AffineMap map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  AffineMap map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  AffineMap map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 2 || map1.getNumResults() != 2 ||
      map2.getNumResults() != 2 || map0.getNumInputs() != 3 ||
      map1.getNumInputs() != 3 || map2.getNumInputs() != 3) {
    return false;
  }

  // Extract dimensions for MxK * NxK -> MxN
  AffineExpr m = map2.getResult(0);
  AffineExpr n = map2.getResult(1);
  AffineExpr k = map0.getResult(1);
  auto *context = indexingMaps.getContext();
  auto mapA = AffineMapAttr::get(AffineMap::get(3, 0, {m, k}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(3, 0, {n, k}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(3, 0, {m, n}, context));
  auto maps = ArrayAttr::get(context, {mapA, mapB, mapC});
  return indexingMaps == maps;
}

/// Sets the lowering configuration for a generic op implementing a
/// transposition.
static LogicalResult setTransposeLikeOpRootConfig(
    mlir::FunctionOpInterface entryPointFn, linalg::LinalgOp linalgOp,
    AIEPassPipeline usePassPipeline, AIEConfig cfg) {
  if (usePassPipeline == AIEPassPipeline::PackPeelPipeline)
    return setRootConfigForPackPeelPipeline(entryPointFn, linalgOp, cfg);
  else if (usePassPipeline == AIEPassPipeline::PadPackPipeline)
    return setRootConfigForPadPackPipeline(entryPointFn, linalgOp, cfg);
  return linalgOp.emitOpError("unhandled pass pipeline");
}

static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   linalg::GenericOp genericOp,
                                   AIEPassPipeline usePassPipeline,
                                   AIEConfig cfg) {
  assert(!getLoweringConfig(genericOp) &&
         "expected lowering_config is not set");

  if (isMatmulTranspose(genericOp) &&
      succeeded(setTransposeLikeOpRootConfig(entryPointFn, genericOp,
                                             usePassPipeline, cfg))) {
    return success();
  }

  return failure();
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   linalg::ContractionOpInterface contractionOp,
                                   AIEPassPipeline usePassPipeline,
                                   AIEConfig cfg) {
  assert(!getLoweringConfig(contractionOp) &&
         "expected lowering_config is not set");
  auto linalgOp = cast<linalg::LinalgOp>(contractionOp.getOperation());
  if (isa<linalg::MatmulTransposeBOp>(linalgOp)) {
    if (succeeded(setTransposeLikeOpRootConfig(entryPointFn, linalgOp,
                                               usePassPipeline, cfg))) {
      return success();
    }
    return failure();
  }
  unsigned numLoops = linalgOp.getNumLoops();
  {
    SmallVector<unsigned> dims;
    linalgOp.getReductionDims(dims);
    if (dims.size() != 1 || dims[0] != numLoops - 1) {
      return linalgOp.emitOpError(
          "expected to have exactly one reduction dim, and it is the innermost "
          "dim");
    }
  }

  // TODO (nmeshram) : This needs to be moved in a separate more generalized
  // logic. Also, need a flag to experiment between pad based and pack based
  // approach which will have different tile sizes and pass pipelines
  if (usePassPipeline == AIEPassPipeline::PackPeelPipeline)
    return setRootConfigForPackPeelPipeline(entryPointFn, linalgOp, cfg);
  if (usePassPipeline == AIEPassPipeline::PadPackPipeline)
    return setRootConfigForPadPackPipeline(entryPointFn, linalgOp, cfg);
  return linalgOp.emitOpError("unhandled pass pipeline");
}

/// Redirects to methods that set the configuration based on operation type.
static LogicalResult setRootConfigImpl(mlir::FunctionOpInterface entryPointFn,
                                       Operation *op,
                                       AIEPassPipeline usePassPipeline,
                                       AIEConfig cfg) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        // TODO (nmeshram): This is very limited for now, plan is to
        // let it first crash for all the other ops and then consiously
        // add support for them, this way we can verify our work.
        .Case<linalg::GenericOp>([&](auto op) {
          return setRootConfig(entryPointFn, op, usePassPipeline, cfg);
        })
        .Case<linalg::ContractionOpInterface>([&](auto op) {
          return setRootConfig(entryPointFn, op, usePassPipeline, cfg);
        })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult setTranslationInfoAndRootConfig(
    mlir::FunctionOpInterface entryPointFn, ArrayRef<Operation *> computeOps,
    AIEPassPipeline usePassPipeline, AIEConfig cfg) {
  // Make sure that lowering_config is not preset on any compute ops.
  for (auto computeOp : computeOps) {
    if (getLoweringConfig(computeOp)) return failure();
  }

  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp)) return failure();
  Operation *rootOperation = rootOp.value();

  // TODO (nmeshram): Handle the case with no known root operation.
  if (!rootOperation) {
    return entryPointFn.emitError("Case with no root ops not yet supported.");
  }

  if (failed(setRootConfigImpl(entryPointFn, rootOperation, usePassPipeline,
                               cfg))) {
    return failure();
  }

  // TODO (nmeshram): // Set vector level tile sizes for other operations
  // individually.

  return success();
}

LogicalResult initAIELaunchConfig(FunctionOpInterface funcOp,
                                  AIEPassPipeline usePassPipeline,
                                  AIEConfig cfg) {
  if (getTranslationInfo(funcOp)) {
    return success();
  }

  // TODO (nmeshram): Need a default pipeline for control flow cases.
  if (funcOp.empty() || !llvm::hasSingleElement(funcOp.getFunctionBody())) {
    return funcOp.emitError("control flow not yet supported.");
  }

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  if (failed(setTranslationInfoAndRootConfig(funcOp, computeOps,
                                             usePassPipeline, cfg))) {
    return failure();
  }

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(funcOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

}  // namespace mlir::iree_compiler::AMDAIE
