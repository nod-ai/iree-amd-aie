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
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "kernel-dispatch"

namespace mlir::iree_compiler::AMDAIE {

using detail::findLargestFactor;

namespace {

FailureOr<std::array<uint32_t, 3>> getMatmulInstructionSize(
    linalg::LinalgOp op) {
  auto getElementType = [](Value v) {
    return cast<ShapedType>(v.getType()).getElementType();
  };

  assert(op->getNumResults() > 0 && op->getNumOperands() > 1 &&
         "expected op to have 2+ operands and 1+ results");

  auto elTypeLhs = getElementType(op->getOperand(0));
  auto elTypeRhs = getElementType(op->getOperand(1));
  auto elTypeAcc = getElementType(op->getResult(0));

  return getAIEMatmulInstructionSize(elTypeLhs, elTypeRhs, elTypeAcc);
}

FailureOr<std::array<uint32_t, 3>> getPackedSize(linalg::LinalgOp linalgOp,
                                                 uint64_t M, uint64_t N,
                                                 uint64_t K) {
  // Depending on the operand/result element types, there might be a specific
  // vector instruction size that must be used on AIE. Some types do not have
  // vector instructions, for example if operands are 32-bit types.
  auto maybeInstructionSize = getMatmulInstructionSize(linalgOp);

  // Operand/result element types do not have vector instructions. In this case,
  // try for a packing of 4x4x8, but if the tensor dimensions M, N, and K are
  // not divisible by the instruction size, then use the largest factor of M, N,
  // K that is divisible. Any packing is valid when there is not vectorization.
  if (failed(maybeInstructionSize)) {
    std::array<uint32_t, 3> packedSize;
    packedSize[0] = findLargestFactor(M, 4);
    packedSize[1] = findLargestFactor(N, 4);
    packedSize[2] = findLargestFactor(K, 8);
    return packedSize;
  }

  // Operand/result element types have vector instructions, and a specific
  // vector size which must be used. If the tensor dimensions M, N, K are not
  // divisible by the instruction size, then fail.
  auto instructionSize = maybeInstructionSize.value();
  if (M % instructionSize[0] != 0 || N % instructionSize[1] != 0 ||
      K % instructionSize[2] != 0) {
    return linalgOp.emitOpError(
               "has element types which must target an AIE instruction size "
               "that does not divide M (")
           << M << "), N (" << N << "), or K (" << K
           << "). The instruction size is m = " << instructionSize[0]
           << ", n = " << instructionSize[1] << ", k = " << instructionSize[2]
           << ".";
  }
  return instructionSize;
}

// Container class for the tiling at level 0 (the AIE shared memory) and level 1
// (the AIE core) in the M-, N-, and K-dimensions of a matmul operation, using
// the pad-pack approach to tiling a matmul. Also contains the packing sizes for
// the M, N, and K dimensions. The packing sizes correspond to matmul
// vector-instruction sizes for vectorizable types.
class ParameterSetting {
 public:
  SmallVector<int64_t> getPackSizeL0() const {
    return {m0Pack, n0Pack, k0Pack};
  }
  SmallVector<int64_t> getPackSizeL1() const {
    return {m1Pack, n1Pack, k1Pack};
  }

  static FailureOr<ParameterSetting> create(linalg::LinalgOp linalgOp,
                                            bool isPackPeel);

  uint32_t getM0() const { return M0; }
  uint32_t getN0() const { return N0; }
  uint32_t getK0() const { return K0; }
  uint32_t getM1() const { return M1; }
  uint32_t getN1() const { return N1; }
  uint32_t getK1() const { return K1; }
  uint32_t getM0Pack() const { return m0Pack; }
  uint32_t getN0Pack() const { return n0Pack; }
  uint32_t getK0Pack() const { return k0Pack; }
  uint32_t getM1Pack() const { return m1Pack; }
  uint32_t getN1Pack() const { return n1Pack; }
  uint32_t getK1Pack() const { return k1Pack; }

 private:
  ParameterSetting(uint32_t M0, uint32_t N0, uint32_t K0, uint32_t M1,
                   uint32_t N1, uint32_t K1, uint32_t m0Pack, uint32_t n0Pack,
                   uint32_t k0Pack, uint32_t m1Pack, uint32_t n1Pack,
                   uint32_t k1Pack)
      : M0(M0),
        N0(N0),
        K0(K0),
        M1(M1),
        N1(N1),
        K1(K1),
        m0Pack(m0Pack),
        n0Pack(n0Pack),
        k0Pack(k0Pack),
        m1Pack(m1Pack),
        n1Pack(n1Pack),
        k1Pack(k1Pack) {}

  uint32_t M0;
  uint32_t N0;
  uint32_t K0;
  uint32_t M1;
  uint32_t N1;
  uint32_t K1;
  uint32_t m0Pack;
  uint32_t n0Pack;
  uint32_t k0Pack;
  uint32_t m1Pack;
  uint32_t n1Pack;
  uint32_t k1Pack;
};

FailureOr<ParameterSetting> ParameterSetting::create(linalg::LinalgOp linalgOp,
                                                     bool isPackPeel) {
  auto initType =
      llvm::cast<ShapedType>(linalgOp.getDpsInitOperand(0)->get().getType());
  auto initShape = initType.getShape();

  auto lhsType =
      llvm::cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  auto lhsShape = lhsType.getShape();

  // Shape of the full matmul operation.
  const uint64_t M = initShape[0];
  const uint64_t N = initShape[1];
  const uint64_t K = lhsShape[1];

  // If we are conservative with ensuring that tiles A, B, and C fit at the
  // different memory levels, we should choose the scale factor based
  // on the largest of the types of A, B, and C, which is the type of C, as
  // accumulation always happens in an element type with at least as may bits as
  // the operand types. We currently choose the scale factor based on the
  // element type of the lhs operand, which works with the current workloads (we
  // have not seen OOM errors).
  //
  // Long-term we should use a different approach, one which takes into account
  // the element types of all tensors which need to be allocated in memory
  // simultaneously.
  FailureOr<unsigned> maybeScaleFactor =
      getTilingScaleFactor(lhsType.getElementType());
  if (failed(maybeScaleFactor)) {
    return linalgOp.emitOpError(
        "does not have the expected bitwidth (64, 32, 16, or 8), could not "
        "determine scale factor.");
  }

  unsigned scaleFactor = maybeScaleFactor.value();

  auto maybePackedSize = getPackedSize(linalgOp, M, N, K);
  if (failed(maybePackedSize)) return failure();
  auto [m1Pack, n1Pack, k1Pack] = maybePackedSize.value();

  // The current ad-hoc algorithm for determining the tiling at level 0 and
  // level 1 is as follows:
  //
  // Step 1: Find the largest tiling for M and N on the AIE core, subject to
  // constraints.
  //
  // Step 2: Find the largest tiling for M and N in AIE shared memory,
  // subject to constraints.
  //
  // Tiling in the K-dimension is done differently TODO(newling)
  // document/reconsider.

  if (isPackPeel) {
    // Assume working on a 2x2 AIE array, so the ideal level 1 tile sizes should
    // be (tileM0/2, tileN0/2). Since packing happens before tiling, and an
    // extra step is performed to fuse pack ops into the loops, the adjusted
    // level 1 tile sizes should be (tileM0/2/packedM1, tileN0/2/packedN1).
    auto maxL1Size = 16 * scaleFactor;
    uint32_t M1 = findLargestFactor(M / m1Pack, maxL1Size / m1Pack, m1Pack);
    uint32_t N1 = findLargestFactor(N / n1Pack, maxL1Size / n1Pack, n1Pack);

    auto maxL0Size = 32 * scaleFactor;
    uint32_t M0 = findLargestFactor(M, maxL0Size, m1Pack * M1);
    uint32_t N0 = findLargestFactor(N, maxL0Size, n1Pack * N1);

    // In pack-peel pipeline there is only one level of tiling for K dimension,
    // so set K1 = 0. The packed outer K dimension needs to be 1, so set K0 = 1.
    uint32_t K1 = 0;
    uint32_t K0 = 1;

    uint32_t m0Pack = M0 / 2;
    uint32_t n0Pack = N0 / 2;
    uint32_t k0Pack = findLargestFactor(K, maxL1Size);

    return ParameterSetting{M0,     N0,     K0,     M1,     N1,     K1,
                            m0Pack, n0Pack, k0Pack, m1Pack, n1Pack, k1Pack};
  } else {
    // Assume working on a 4x4 AIE array. The tile sizes are chosen empirically
    // for large GEMM sizes, which are [64*s, 64*s, 256] for the first level and
    // [16*s, 16*s, 16*s] for the second level, where 's' is the scaling factor
    // based on the element type's bit width.
    auto maxL1Size = 16 * scaleFactor;
    uint32_t M1 = findLargestFactor(M, maxL1Size, m1Pack);
    uint32_t N1 = findLargestFactor(N, maxL1Size, n1Pack);

    auto maxL0Size = 64 * scaleFactor;
    uint32_t M0 = findLargestFactor(M, maxL0Size, M1);
    uint32_t N0 = findLargestFactor(N, maxL0Size, N1);

    // Step 3 (currently only for pad-pack pipeline): We assume a 4x4 AIE array.
    // We would like to make use of all the cores. Ideally we'll send 4 distinct
    // horizontal slices of 'A' and 4 distinct vertical slices of 'B' to the AIE
    // array. We check if the tiling is only by a factor of 2, in which case
    // halve the core tiling size so that it divides the memory tile by 4.
    // TODO(newling) in the case of M1:M0 = 1:1 there is a compilation error, so
    // currently only handling the case M1:M0 = 1:2
    //
    // This function either returns coreTile or coreTile/2.
    //
    // It returns coreTile/2 if
    //
    //  1. coreTile is exactly 2x smaller than memTile, and
    //  2. coreTile is divisible by 2, and
    //  3. coreTile is divisible by pack.
    //
    // Otherwise, it returns coreTile.
    auto downScale = [](uint32_t pack, uint64_t memTile, uint64_t coreTile) {
      // If coreTile is not exactly 2x smaller than memTile, return coreTile
      // (don't downscale).
      if (memTile / coreTile != 2) return coreTile;
      if (memTile % coreTile != 0) return coreTile;

      // If coreTile cannot be divided by 2, return coreTile (don't downscale).
      if (coreTile % 2 != 0) return coreTile;

      // The core tile size must still be a multiple of the packing size.
      auto reduced = coreTile / 2;
      if (reduced % pack != 0) return coreTile;

      return reduced;
    };
    M1 = downScale(m1Pack, M0, M1);
    N1 = downScale(n1Pack, N0, N1);

    uint32_t K1 = findLargestFactor(K / k1Pack, 2 * scaleFactor);
    uint32_t K0 = findLargestFactor(K, 256, k1Pack * K1);

    // In pad-pack pipeline, there is only one level of packing, set pack
    // parameters for level 0 as 0.
    uint32_t m0Pack = 0;
    uint32_t n0Pack = 0;
    uint32_t k0Pack = 0;

    // Tiling at level 0 (shared memory) should be no smaller than tiling at
    // level 1 (AIE core memory).
    assert(M0 >= M1 && N0 >= N1);

    // Tiling at level 1 (AIE core memory) should be no smaller than the packing
    // size (vector instruction size).
    assert(M1 >= m1Pack && N1 >= n1Pack);

    return ParameterSetting{M0,     N0,     K0,     M1,     N1,     K1,
                            m0Pack, n0Pack, k0Pack, m1Pack, n1Pack, k1Pack};
  }
}
}  // namespace

static SmallVector<int64_t> setInnerPermB(bool isMatmulTransposeB) {
  SmallVector<int64_t> innerPermB;
  if (isMatmulTransposeB) {
    innerPermB = {0, 1};
  } else {
    innerPermB = {1, 0};
  }
  return innerPermB;
}

static LogicalResult setRootConfigForPackPeelPipeline(
    mlir::FunctionOpInterface entryPointFn, linalg::LinalgOp linalgOp,
    AIEConfig cfg, bool isMatmulTransposeB) {
  auto maybePackPeelTiling = ParameterSetting::create(linalgOp, true);
  if (failed(maybePackPeelTiling)) return failure();
  auto packPeelTiling = maybePackPeelTiling.value();

  // ------------------------------------------------------
  // --------------- Set packing config -------------------
  // ------------------------------------------------------
  MLIRContext *context = entryPointFn.getContext();

  // For matmul, transpose B matrix from [K N n k] to [K N k n]
  // For matmul_transpose_b, we don't have to transpose the B matrix,
  // since it is already [N K n k]
  SmallVector<int64_t> transposePackIndices = {1};
  // There is no corresponding unpack for the specified pack operation
  // 0 is used when unpack is empty
  SmallVector<bool> unpackEmpty = {false};
  SmallVector<int64_t> innerPermB = setInnerPermB(isMatmulTransposeB);
  SmallVector<SmallVector<int64_t>> innerPerm = {innerPermB};
  SmallVector<SmallVector<int64_t>> outerPerm = {{0, 1}};
  auto packingConfigLevel1Attr = getPackingConfigPackingLevelAttr(
      context, packPeelTiling.getPackSizeL0(), transposePackIndices,
      unpackEmpty, innerPerm, outerPerm);

  // Pack level => 2.
  // packed size for [M, N, K, m, n, k]
  SmallVector<int64_t> packedSizes = {0,
                                      0,
                                      0,
                                      packPeelTiling.getM1Pack(),
                                      packPeelTiling.getN1Pack(),
                                      packPeelTiling.getK1Pack()};

  // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
  // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
  // For matmul, transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
  // For matmul_transpose_b, transpose B matrix from [N K n k n0 k0] to
  // [N K k n n0 k0]
  transposePackIndices = {0, 1, 2};
  // Only the third pack operation has a corresponding unpack operation
  unpackEmpty = {false, false, true};
  innerPerm = {{0, 1}, innerPermB, {0, 1}};
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
  SmallVector<int64_t> TileSizeLevel0 = {packPeelTiling.getM0(),
                                         packPeelTiling.getN0()};
  SmallVector<int64_t> TileSizeLevel1 = {0, 0, packPeelTiling.getK0()};
  SmallVector<int64_t> TileSizeLevel2 = {1, 1, 0, 0, 0, 0};
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
    AIEConfig cfg, bool isMatmulTransposeB) {
  auto maybePadPackTiling = ParameterSetting::create(linalgOp, false);
  if (failed(maybePadPackTiling)) return failure();
  auto padPackTiling = maybePadPackTiling.value();

  // Do packing first to allow better packing configs
  // ------------------------------------------------------
  // --------------- Set packing config -------------------
  // ------------------------------------------------------
  MLIRContext *context = entryPointFn.getContext();

  // Transpose A matrix from [M K m k] to [K M m k]
  // Transpose C matrix from [M N m n] to [N M m n]
  // For matmul, transpose B matrix from [K N n k] to [N K k n]
  // For matmul_transpose_b, transpose B matrix from [N K n k] to [K N n k]
  SmallVector<int64_t> transposePackIndices{0, 1, 2};
  SmallVector<bool> unpackEmpty{false, false, true};
  SmallVector<int64_t> innerPermB = setInnerPermB(isMatmulTransposeB);
  SmallVector<SmallVector<int64_t>> innerPerm{{0, 1}, innerPermB, {0, 1}};
  SmallVector<SmallVector<int64_t>> outerPerm{{1, 0}, {1, 0}, {1, 0}};

  auto packingConfigLevel1Attr = getPackingConfigPackingLevelAttr(
      context, padPackTiling.getPackSizeL1(), transposePackIndices, unpackEmpty,
      innerPerm, outerPerm);
  SmallVector<PackingConfigPackingLevelAttr> packingConfigLevelsVal{
      packingConfigLevel1Attr};

  auto packingConfigLevels =
      PackingConfigPackingLevelsAttr::get(context, packingConfigLevelsVal);
  auto config = PackingConfigAttr::get(context, packingConfigLevels);
  setPackingConfig(linalgOp, config);

  // ------------------------------------------------------
  // -------------- Set lowering config -------------------
  // ------------------------------------------------------
  SmallVector<int64_t> level0{padPackTiling.getM0(), padPackTiling.getN0()};
  SmallVector<int64_t> level1{0, 0, padPackTiling.getK0()};
  SmallVector<int64_t> level2{padPackTiling.getM1(), padPackTiling.getN1()};
  SmallVector<int64_t> level3{0, 0, padPackTiling.getK1()};
  TileSizesListType tileSizes = {level0, level1, level2, level3};

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
  auto lhsBlockArg = dyn_cast<BlockArgument>(mulOp->getOperand(0));
  auto rhsBlockArg = dyn_cast<BlockArgument>(mulOp->getOperand(1));
  auto outBlockArg = dyn_cast<BlockArgument>(addOp->getOperand(0));
  if (!lhsBlockArg || !rhsBlockArg || !outBlockArg ||
      lhsBlockArg.getOwner() != body || rhsBlockArg.getOwner() != body ||
      outBlockArg.getOwner() != body || lhsBlockArg.getArgNumber() != 0 ||
      rhsBlockArg.getArgNumber() != 1 || outBlockArg.getArgNumber() != 2) {
    return false;
  }
  return true;
}

/// `isMatmulTransposeB` is a utility function that aims to indentify whether a
/// linalg.generic op is a matmul with rhs operand transposed.
static bool isMatmulTransposeB(linalg::GenericOp genericOp) {
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
    AIEPassPipeline passPipeline, AIEConfig cfg) {
  if (passPipeline == AIEPassPipeline::PackPeelPipeline)
    return setRootConfigForPackPeelPipeline(entryPointFn, linalgOp, cfg, true);
  else if (passPipeline == AIEPassPipeline::PadPackPipeline)
    return setRootConfigForPadPackPipeline(entryPointFn, linalgOp, cfg, true);
  return linalgOp.emitError(
      "Unhandled pass pipeline in setTransposeLikeOpRootConfig.");
}

static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   linalg::GenericOp genericOp,
                                   AIEPassPipeline passPipeline,
                                   AIEConfig cfg) {
  assert(!getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(genericOp) &&
         "expected lowering_config is not set");

  if (isMatmulTransposeB(genericOp) &&
      succeeded(setTransposeLikeOpRootConfig(entryPointFn, genericOp,
                                             passPipeline, cfg))) {
    return success();
  }

  return failure();
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   linalg::ContractionOpInterface contractionOp,
                                   AIEPassPipeline passPipeline,
                                   AIEConfig cfg) {
  assert(!getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(contractionOp) &&
         "expected lowering_config is not set");
  auto linalgOp = cast<linalg::LinalgOp>(contractionOp.getOperation());
  if (isa<linalg::MatmulTransposeBOp>(linalgOp)) {
    if (succeeded(setTransposeLikeOpRootConfig(entryPointFn, linalgOp,
                                               passPipeline, cfg))) {
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
                 "is expected to have exactly one reduction dim, ")
             << "and that it is the innermost dim (" << numLoops - 1 << ").";
    }
  }

  // TODO (nmeshram) : This needs to be moved in a separate more generalized
  // logic. Also, need a flag to experiment between pad based and pack based
  // approach which will have different tile sizes and pass pipelines
  if (passPipeline == AIEPassPipeline::PackPeelPipeline)
    return setRootConfigForPackPeelPipeline(entryPointFn, linalgOp, cfg, false);
  if (passPipeline == AIEPassPipeline::PadPackPipeline)
    return setRootConfigForPadPackPipeline(entryPointFn, linalgOp, cfg, false);
  return linalgOp.emitError("Unhandled pass pipeline in setRootConfig.");
}

/// Redirects to methods that set the configuration based on operation type.
static LogicalResult setRootConfigImpl(mlir::FunctionOpInterface entryPointFn,
                                       Operation *op,
                                       AIEPassPipeline passPipeline,
                                       AIEConfig cfg) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        // TODO (nmeshram): This is very limited for now, plan is to
        // let it first crash for all the other ops and then consiously
        // add support for them, this way we can verify our work.
        .Case<linalg::GenericOp>([&](auto op) {
          return setRootConfig(entryPointFn, op, passPipeline, cfg);
        })
        .Case<linalg::ContractionOpInterface>([&](auto op) {
          return setRootConfig(entryPointFn, op, passPipeline, cfg);
        })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult setTranslationInfoAndRootConfig(
    mlir::FunctionOpInterface entryPointFn, ArrayRef<Operation *> computeOps,
    AIEPassPipeline passPipeline, AIEConfig cfg) {
  // Make sure that lowering_config is not preset on any compute ops.
  for (auto computeOp : computeOps) {
    if (getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(computeOp))
      return failure();
  }

  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp)) return failure();
  Operation *rootOperation = rootOp.value();

  // TODO (nmeshram): Handle the case with no known root operation.
  if (!rootOperation)
    return entryPointFn.emitError("Case with no root ops not yet supported.");

  if (failed(setRootConfigImpl(entryPointFn, rootOperation, passPipeline, cfg)))
    return failure();
  return success();
}

LogicalResult initAIELaunchConfig(FunctionOpInterface funcOp,
                                  AIEPassPipeline passPipeline, AIEConfig cfg) {
  if (getTranslationInfo(funcOp)) return success();

  // TODO (nmeshram): Need a default pipeline for control flow cases.
  if (funcOp.empty() || !llvm::hasSingleElement(funcOp.getFunctionBody()))
    return funcOp.emitError("Control flow not yet supported.");

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  if (failed(setTranslationInfoAndRootConfig(funcOp, computeOps, passPipeline,
                                             cfg)))
    return failure();

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(funcOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

}  // namespace mlir::iree_compiler::AMDAIE
