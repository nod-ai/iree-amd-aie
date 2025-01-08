// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/KernelDispatch.h"

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "kernel-dispatch"

namespace mlir::iree_compiler::AMDAIE {

using detail::findLargestFactor;

//===----------------------------------------------------------------------===//
// Config Setting Helpers
//===----------------------------------------------------------------------===//

namespace {

FailureOr<std::array<uint32_t, 3>> getMatmulInstructionSize(
    linalg::LinalgOp op, AMDAIEDevice targetDevice) {
  auto getElementType = [](Value v) {
    return cast<ShapedType>(v.getType()).getElementType();
  };

  assert(op->getNumResults() > 0 && op->getNumOperands() > 1 &&
         "expected op to have 2+ operands and 1+ results");

  auto elTypeLhs = getElementType(op->getOperand(0));
  auto elTypeRhs = getElementType(op->getOperand(1));
  auto elTypeAcc = getElementType(op->getResult(0));

  AMDAIEDeviceModel deviceModel = AMDAIE::getDeviceModel(targetDevice);
  return deviceModel.getAIEMatmulInstructionSize(elTypeLhs, elTypeRhs,
                                                 elTypeAcc);
}

FailureOr<std::array<uint32_t, 3>> getPackedSize(linalg::LinalgOp linalgOp,
                                                 uint64_t M, uint64_t N,
                                                 uint64_t K,
                                                 AMDAIEDevice targetDevice) {
  // Depending on the operand/result element types, there might be a specific
  // vector instruction size that must be used on AIE. Some types do not have
  // vector instructions, for example if operands are 32-bit types.
  FailureOr<std::array<uint32_t, 3>> maybeInstructionSize =
      getMatmulInstructionSize(linalgOp, targetDevice);

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
                                            bool isPackPeel, bool isObjectFifo,
                                            AMDAIEDevice targetDevice,
                                            uint32_t numRows, uint32_t numCols);

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

FailureOr<ParameterSetting> ParameterSetting::create(
    linalg::LinalgOp linalgOp, bool isPackPeel, bool isObjectFifo,
    AMDAIEDevice targetDevice, uint32_t numRows, uint32_t numCols) {
  auto initType =
      llvm::cast<ShapedType>(linalgOp.getDpsInitOperand(0)->get().getType());
  ArrayRef<int64_t> initShape = initType.getShape();

  auto lhsType =
      llvm::cast<ShapedType>(linalgOp.getDpsInputOperand(0)->get().getType());
  ArrayRef<int64_t> lhsShape = lhsType.getShape();

  // Shape of the full matmul operation.
  if (isa<linalg::BatchMatmulOp>(linalgOp)) {
    initShape = initShape.drop_front();
    lhsShape = lhsShape.drop_front();
  }
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
      isObjectFifo ? getTilingScaleFactor(initType.getElementType())
                   : getTilingScaleFactor(lhsType.getElementType());
  if (failed(maybeScaleFactor)) {
    return linalgOp.emitOpError(
        "does not have the expected bitwidth (64, 32, 16, or 8), could not "
        "determine scale factor.");
  }

  unsigned scaleFactor = maybeScaleFactor.value();

  auto maybePackedSize = getPackedSize(linalgOp, M, N, K, targetDevice);
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
    // Assume working on a (numRows, numCols) AIE array, so the ideal level 1
    // tile sizes should be (tileM0/numRows, tileN0/numCols). Since packing
    // happens before tiling, and an extra step is performed to fuse pack ops
    // into the loops, the adjusted level 1 tile sizes should be
    // (tileM0/numRows/packedM1, tileN0/numCols/packedN1).
    // TODO (vivian): Refactor the codes with the following aspect:
    // Develop a better way to select tile sizes to make the most use of
    // memory while taking all factors (double buffer, elementwise memory usage,
    // lhs/rhs element type, etc) into account.
    uint32_t maxL1SizeM = 16 * scaleFactor;
    uint32_t maxL1SizeN = 16 * scaleFactor;
    uint32_t M1 = findLargestFactor(M / m1Pack, maxL1SizeM / m1Pack, m1Pack);
    uint32_t N1 = findLargestFactor(N / n1Pack, maxL1SizeN / n1Pack, n1Pack);

    uint32_t maxL0SizeM = numRows * maxL1SizeM;
    uint32_t maxL0SizeN = numCols * maxL1SizeM;
    uint32_t M0 = findLargestFactor(M, maxL0SizeM, m1Pack * M1);
    uint32_t N0 = findLargestFactor(N, maxL0SizeN, n1Pack * N1);

    // In pack-peel pipeline there is only one level of tiling for K dimension,
    // so set K1 = 0. The packed outer K dimension needs to be 1, so set K0 = 1.
    uint32_t K1 = 0;
    uint32_t K0 = 1;
    uint32_t maxL1SizeK = 16 * scaleFactor;
    uint32_t k0Pack = findLargestFactor(K, maxL1SizeK);

    // Instead of directly packing to (1, 1, M0, N0), the new strategy is making
    // the pack size as (numRows, numCols, M0/numRows, N0/numCols) to avoid the
    // large allocation in L1. Also we should make sure the first level inner
    // pack size is divisible by the second level of inner pack size (vector
    // instruction size).
    uint32_t m0Pack = (M0 / numRows) % m1Pack == 0 ? (M0 / numRows) : M0;
    uint32_t n0Pack = (N0 / numCols) % n1Pack == 0 ? (N0 / numCols) : N0;

    return ParameterSetting{M0,     N0,     K0,     M1,     N1,     K1,
                            m0Pack, n0Pack, k0Pack, m1Pack, n1Pack, k1Pack};
  } else {
    // Assume working on a (numRows, numCols) AIE array. The tile sizes are
    // chosen empirically for large GEMM sizes, which are [64*s, 64*s, 256] for
    // the first level and [16*s, 16*s, 16*s] for the second level, where 's' is
    // the scaling factor based on the element type's bit width.
    uint32_t maxL1SizeM = 16 * scaleFactor;
    uint32_t maxL1SizeN = 16 * scaleFactor;
    uint32_t M1 = findLargestFactor(M, maxL1SizeM, m1Pack);
    uint32_t N1 = findLargestFactor(N, maxL1SizeN, n1Pack);

    uint32_t maxL0SizeM = numRows * maxL1SizeM;
    uint32_t maxL0SizeN = numCols * maxL1SizeM;
    uint32_t M0 = findLargestFactor(M, maxL0SizeM, M1);
    uint32_t N0 = findLargestFactor(N, maxL0SizeN, N1);

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
  SmallVector<int64_t> innerPerm;
  if (isMatmulTransposeB) {
    innerPerm = {0, 1};
  } else {
    innerPerm = {1, 0};
  }
  return innerPerm;
}

static SmallVector<int64_t> setInnerPermA(bool isMatmulTransposeA) {
  SmallVector<int64_t> innerPerm;
  if (isMatmulTransposeA) {
    innerPerm = {1, 0};
  } else {
    innerPerm = {0, 1};
  }
  return innerPerm;
}

//===----------------------------------------------------------------------===//
// Configuration for Matmul Pipelines
//===----------------------------------------------------------------------===//

static LogicalResult setRootConfigForPackPeelPipeline(
    mlir::FunctionOpInterface entryPointFn, linalg::LinalgOp linalgOp,
    LowerToAIEPassPipeline useLowerToAIEPipeline, AMDAIEDevice targetDevice,
    uint32_t numRows, uint32_t numCols) {
  bool isObjectFifo =
      useLowerToAIEPipeline == LowerToAIEPassPipeline::ObjectFifo;
  auto maybePackPeelTiling =
      ParameterSetting::create(linalgOp, /*isPackPeel=*/true, isObjectFifo,
                               targetDevice, numRows, numCols);
  if (failed(maybePackPeelTiling)) return failure();
  auto packPeelTiling = maybePackPeelTiling.value();

  // ------------------------------------------------------
  // --------------- Set packing config -------------------
  // ------------------------------------------------------
  MLIRContext *context = entryPointFn.getContext();

  SmallVector<int64_t> packedSizesL0 = packPeelTiling.getPackSizeL0();
  if (isa<linalg::BatchMatmulOp>(linalgOp)) {
    packedSizesL0.insert(packedSizesL0.begin(), 0);
  }

  // For matmul, transpose B matrix from [K N n k] to [K N k n]
  // For matmul_transpose_b, we don't have to transpose the B matrix,
  // since it is already [N K n k]
  SmallVector<int64_t> transposePackIndices = {0, 1};
  // There is no corresponding unpack for the specified pack operation
  // 0 is used when unpack is empty
  SmallVector<bool> unpackEmpty = {false, false};
  SmallVector<int64_t> innerPermA = setInnerPermA(isMatmulTransposeA(linalgOp));
  SmallVector<int64_t> innerPermB = setInnerPermB(isMatmulTransposeB(linalgOp));
  SmallVector<SmallVector<int64_t>> innerPerm = {innerPermA, innerPermB};
  SmallVector<int64_t> outerPermVec = {0, 1};
  if (isa<linalg::BatchMatmulOp>(linalgOp)) {
    outerPermVec.push_back(2);
  }
  SmallVector<SmallVector<int64_t>> outerPerm = {outerPermVec, outerPermVec};
  if (isObjectFifo) {
    // Add outer permutation for unpack. NOTE: This currently fails for some
    // tests in the AIR pipeline.
    transposePackIndices.push_back(2);
    unpackEmpty.push_back(true);
    innerPerm.push_back({0, 1});
    if (isa<linalg::BatchMatmulOp>(linalgOp)) {
      outerPerm.push_back({0, 2, 1});
    } else {
      outerPerm.push_back({1, 0});
    }
  }

  auto packingConfigLevel0Attr = getPackingConfigPackingLevelAttr(
      context, packedSizesL0, transposePackIndices, unpackEmpty, innerPerm,
      outerPerm);

  // Pack level => 2.
  // packed size for [M, N, K, m, n, k]
  SmallVector<int64_t> packedSizesL1 = {0,
                                        0,
                                        0,
                                        packPeelTiling.getM1Pack(),
                                        packPeelTiling.getN1Pack(),
                                        packPeelTiling.getK1Pack()};

  if (isa<linalg::BatchMatmulOp>(linalgOp)) {
    packedSizesL1.insert(packedSizesL1.begin(), 0);
  }

  // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
  // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
  // For matmul, transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
  // For matmul_transpose_b, transpose B matrix from [N K n k n0 k0] to
  // [N K k n n0 k0]
  transposePackIndices = {0, 1, 2};
  // Only the third pack operation has a corresponding unpack operation
  unpackEmpty = {false, false, true};
  innerPerm = {innerPermA, innerPermB, {0, 1}};
  if (isa<linalg::BatchMatmulOp>(linalgOp)) {
    outerPerm = {{0, 1, 2, 4, 3}, {0, 1, 2, 4, 3}, {0, 1, 2, 4, 3}};
  } else {
    outerPerm = {{0, 1, 3, 2}, {0, 1, 3, 2}, {0, 1, 3, 2}};
  }
  auto packingConfigLevel1Attr = getPackingConfigPackingLevelAttr(
      context, packedSizesL1, transposePackIndices, unpackEmpty, innerPerm,
      outerPerm);

  SmallVector<PackingConfigPackingLevelAttr> packingConfigLevelsVal = {
      packingConfigLevel0Attr, packingConfigLevel1Attr};
  auto packingConfigLevels =
      PackingConfigPackingLevelsAttr::get(context, packingConfigLevelsVal);
  auto config = PackingConfigAttr::get(context, packingConfigLevels);
  setPackingConfig(linalgOp, config);

  // ------------------------------------------------------
  // -------------- Set lowering config -------------------
  // ------------------------------------------------------
  SmallVector<int64_t> tileSizeLevel0 = {packPeelTiling.getM0(),
                                         packPeelTiling.getN0()};
  SmallVector<int64_t> tileSizeLevel1 = {0, 0, packPeelTiling.getK0()};
  SmallVector<int64_t> tileSizeLevel2 = {1, 1, 0, 0, 0, 0};

  if (isa<linalg::BatchMatmulOp>(linalgOp)) {
    tileSizeLevel0.insert(tileSizeLevel0.begin(), 1);
    tileSizeLevel1.insert(tileSizeLevel1.begin(), 0);
    tileSizeLevel2.insert(tileSizeLevel2.begin(), 0);
  }

  TileSizesListType tileSizes = {tileSizeLevel0, tileSizeLevel1,
                                 tileSizeLevel2};
  if (failed(setOpConfigAndEntryPointFnTranslation(
          entryPointFn, linalgOp, tileSizes,
          IREE::Codegen::DispatchLoweringPassPipeline::Custom))) {
    return failure();
  }
  return success();
}

static LogicalResult setRootConfigForPadPackPipeline(
    mlir::FunctionOpInterface entryPointFn, linalg::LinalgOp linalgOp,
    AMDAIEDevice targetDevice, uint32_t numRows, uint32_t numCols) {
  auto maybePadPackTiling = ParameterSetting::create(
      linalgOp, /*isPackPeel=*/false, /*isObjectFifo=*/false, targetDevice,
      numRows, numCols);
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
  SmallVector<int64_t> innerPermA = setInnerPermA(isMatmulTransposeA(linalgOp));
  SmallVector<int64_t> innerPermB = setInnerPermB(isMatmulTransposeB(linalgOp));
  SmallVector<SmallVector<int64_t>> innerPerm{innerPermA, innerPermB, {0, 1}};
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

//===----------------------------------------------------------------------===//
// Configuration for Convolution Pipelines
//===----------------------------------------------------------------------===//

static LogicalResult setRootConfigForConvDecomposePipeline(
    mlir::FunctionOpInterface entryPointFn, linalg::LinalgOp linalgOp,
    AMDAIEDevice targetDevice) {
  MLIRContext *context = entryPointFn.getContext();

  FailureOr<std::array<uint32_t, 3>> maybeInstructionSize =
      getMatmulInstructionSize(linalgOp, targetDevice);
  int64_t OW = 4;
  int64_t OC = 4;
  int64_t IC = 8;
  if (succeeded(maybeInstructionSize)) {
    auto [m, n, k] = maybeInstructionSize.value();
    OW = m;
    OC = n;
    IC = k;
  }

  SmallVector<int64_t> transposePackIndices{0, 1, 2};
  SmallVector<bool> unpackEmpty{false, false, true};

  // Convolution type specific vectors:
  SmallVector<SmallVector<int64_t>> innerPerm;
  SmallVector<SmallVector<int64_t>> outerPerm;
  SmallVector<int64_t> tileSizeLevel0;
  SmallVector<int64_t> tileSizeLevel1;
  SmallVector<int64_t> tileSizeLevel2;
  SmallVector<int64_t> packingSizes;

  // [N, OH, OW, OC, KH, KW, IC].
  if (isa<linalg::Conv2DNhwcHwcfQOp>(linalgOp) ||
      isa<linalg::Conv2DNhwcHwcfOp>(linalgOp)) {
    // The goal is to pack the input image and kernel as follows, when moving
    // from L2 to L1 (example where there are 32 input channels):
    // Image: memref<1x3x6x32xbf16> ->  memref<1x3x4x6x8xbf16>
    // Kernel: memref<3x3x32x4xbf16> -> memref<3x3x4x1x8x4xbf16>
    innerPerm = {{}, {{1, 0}}, {}};
    outerPerm = {{0, 1, 3, 2}, {}, {0, 1, 2, 3}};
    packingSizes = {0, 0, 0, OC, 0, 0, IC};
    // Target one column of 4 cores, each core processing a different
    // output image row. TODO(newling) use 4x4 array.
    // https://github.com/nod-ai/iree-amd-aie/issues/821
    tileSizeLevel0 = {1, 4, OW, OC, 0, 0, 0};
    tileSizeLevel1 = {1, 1, OW, OC, 0, 0, 0};
    // scf.for tiling of KH, KW, and (packed) IC dimensions:
    tileSizeLevel2 = {0, 0, 0, 0, 1, 1, 1, 0, 0};
  }

  // [N, OC, OH, OW, IC, KH, KW]
  else if (isa<linalg::Conv2DNchwFchwOp>(linalgOp)) {
    // The matmul reduction dimension is the input channel (IC) dimension.
    // For Conv2DNhwcHwcfOp, this dimension is already the inner-most dimension
    // of the input image, and the penultimate dimension of the kernel --
    // exactly what we want. For Conv2DNchwFchwOp, can the tensor dimensions be
    // permuted in DMA to get them in the correct positions? For the image
    // tensor, only if H*W is a nice power of 2 (DMA constraint). For kernels,
    // it requires h*w is a nice power of 2 -- unlikely, we typically have
    // h=w=3. The dimension permutations will therefore often therefore need to
    // be done on the core. We leave this for future work, the expectation for
    // now is that models have been transformed at a high level to avoid
    // channel-first convolutions.
    return linalgOp.emitError(
        "Only channel-last convolution supported currently.");
  }

  // [N, OH, OW, C, KW, HW]
  else if (isa<linalg::DepthwiseConv2DNhwcHwcOp>(linalgOp)) {
    // Notes
    // =====
    // A property of depthwise convolution is that it can't be expressed in
    // terms of matmul, unlike the above (dense) conv-2ds. The tile sizes we
    // choose below are therefore not constrained by AIE matmul instructions.
    //
    // The logic is currently fragile, and there are no guardrails: there are
    // no checks that the data tiles are not too large, or that the input
    // dimensions are perfectly tiled by the hard-coded tile dimensions below.
    // These will be done as a follow-up task.
    auto getElementType = [](Value v) {
      return cast<ShapedType>(v.getType()).getElementType();
    };
    const uint16_t OW_0 = 4;
    const uint16_t OH_1 = 1;

    auto operandType = getElementType(linalgOp->getOperand(0));
    auto maybeMacNumElements = getAIEMacNumElements(
        operandType, getElementType(linalgOp->getResult(0)));
    uint16_t OC_0 = 16;
    if (!failed(maybeMacNumElements)) {
      OC_0 = maybeMacNumElements.value();
    }
    // If the operand type has fewer than 32-bits, we really should be able to
    // get a mac-width for it. Bail because we didn't, there's probably just
    // something missing in a table.
    else if (operandType.getIntOrFloatBitWidth() < 32) {
      return linalgOp.emitError(
          "has an operand type with fewer than 32-bits, but no mac-width "
          "could be determined.");
    }

    const uint16_t OC_1 = OC_0 / 4;
    packingSizes = {0, 0, 0, OC_1, 0, 0};
    innerPerm = {{}, {}, {}};
    outerPerm = {{0, 1, 2, 3}, {0, 1, 2}, {0, 1, 2, 3}};
    // Target one column of 4 cores, each core processing a different
    // output image row. TODO(newling) use 4x4 array.
    // https://github.com/nod-ai/iree-amd-aie/issues/821
    tileSizeLevel0 = {1, 4 * OH_1, OW_0, OC_1, 0, 0};
    tileSizeLevel1 = {1, OH_1, OW_0, OC_1, 0, 0};
    tileSizeLevel2 = {0, 0, 0, 0, 1, 1, 0};
  }

  else {
    return linalgOp.emitError(
        "unrecognised convolution op, cannot set packing config. ");
  }

  assert(!innerPerm.empty() && !outerPerm.empty() && !packingSizes.empty() &&
         !tileSizeLevel0.empty() && !tileSizeLevel1.empty() &&
         "not all vectors for initializing config are non-empty");

  auto packingConfigLevel1Attr = getPackingConfigPackingLevelAttr(
      context, packingSizes, transposePackIndices, unpackEmpty, innerPerm,
      outerPerm);
  SmallVector<PackingConfigPackingLevelAttr> packingConfigLevelsVal{
      packingConfigLevel1Attr};

  auto packingConfigLevels =
      PackingConfigPackingLevelsAttr::get(context, packingConfigLevelsVal);
  auto config = PackingConfigAttr::get(context, packingConfigLevels);
  setPackingConfig(linalgOp, config);

  TileSizesListType tileSizes = {tileSizeLevel0, tileSizeLevel1,
                                 tileSizeLevel2};

  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, linalgOp, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::Custom);
}

//===----------------------------------------------------------------------===//
// Root Configurations
//===----------------------------------------------------------------------===//

/// Sets the lowering configuration for dispatch region with root op that
/// is a generic op.
static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   linalg::GenericOp genericOp,
                                   TilePassPipeline passPipeline,
                                   LowerToAIEPassPipeline useLowerToAIEPipeline,
                                   AMDAIEDevice targetDevice, uint32_t numRows,
                                   uint32_t numCols) {
  assert(!getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(genericOp) &&
         "expected lowering_config is not set");
  if (!isMatmul(genericOp) && !isMatmulTransposeA(genericOp) &&
      !isMatmulTransposeB(genericOp))
    return genericOp.emitOpError(
        "Current pipelines are only set for matmul-like ops.");

  if (passPipeline == TilePassPipeline::PackPeelPipeline)
    return setRootConfigForPackPeelPipeline(entryPointFn, genericOp,
                                            useLowerToAIEPipeline, targetDevice,
                                            numRows, numCols);
  if (passPipeline == TilePassPipeline::PadPackPipeline)
    return setRootConfigForPadPackPipeline(entryPointFn, genericOp,
                                           targetDevice, numRows, numCols);
  return genericOp.emitError("Unhandled pass pipeline in setRootConfig.");
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   linalg::ContractionOpInterface contractionOp,
                                   TilePassPipeline passPipeline,
                                   LowerToAIEPassPipeline useLowerToAIEPipeline,
                                   AMDAIEDevice targetDevice, uint32_t numRows,
                                   uint32_t numCols) {
  assert(!getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(contractionOp) &&
         "expected lowering_config is not set");
  auto linalgOp = cast<linalg::LinalgOp>(contractionOp.getOperation());
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
  if (passPipeline == TilePassPipeline::PackPeelPipeline)
    return setRootConfigForPackPeelPipeline(entryPointFn, linalgOp,
                                            useLowerToAIEPipeline, targetDevice,
                                            numRows, numCols);
  if (passPipeline == TilePassPipeline::PadPackPipeline)
    return setRootConfigForPadPackPipeline(entryPointFn, linalgOp, targetDevice,
                                           numRows, numCols);
  return linalgOp.emitError("Unhandled pass pipeline in setRootConfig.");
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the convolution operation interface.
static LogicalResult setConvRootConfig(mlir::FunctionOpInterface entryPointFn,
                                       linalg::ConvolutionOpInterface convOp,
                                       TilePassPipeline passPipeline,
                                       AMDAIEDevice targetDevice) {
  assert(!getLoweringConfig<IREE::Codegen::LoweringConfigAttr>(convOp) &&
         "expected lowering_config is not set");
  auto linalgOp = cast<linalg::LinalgOp>(convOp.getOperation());

  // Current tiling strategy is based on llvm-cpu ConvTileAndDecomposeExpert.
  if (passPipeline == TilePassPipeline::ConvDecomposePipeline)
    return setRootConfigForConvDecomposePipeline(entryPointFn, linalgOp,
                                                 targetDevice);
  return linalgOp.emitError("Unhandled pass pipeline in setConvRootConfig.");
}

/// Redirects to methods that set the configuration based on operation type.
static LogicalResult setRootConfigImpl(
    mlir::FunctionOpInterface entryPointFn, Operation *op,
    TilePassPipeline passPipeline, LowerToAIEPassPipeline useLowerToAIEPipeline,
    AMDAIEDevice targetDevice, uint32_t numRows, uint32_t numCols) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        // TODO (nmeshram): This is very limited for now, plan is to
        // let it first crash for all the other ops and then consiously
        // add support for them, this way we can verify our work.
        // TODO (vivian): add support for other conv interface ops
        .Case<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp,
              linalg::Conv2DNhwcHwcfQOp, linalg::DepthwiseConv2DNhwcHwcOp>(
            [&](auto op) {
              return setConvRootConfig(entryPointFn, op, passPipeline,
                                       targetDevice);
            })
        .Case<linalg::GenericOp>([&](auto op) {
          return setRootConfig(entryPointFn, op, passPipeline,
                               useLowerToAIEPipeline, targetDevice, numRows,
                               numCols);
        })
        .Case<linalg::ContractionOpInterface>([&](auto op) {
          return setRootConfig(entryPointFn, op, passPipeline,
                               useLowerToAIEPipeline, targetDevice, numRows,
                               numCols);
        })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult setTranslationInfoAndRootConfig(
    mlir::FunctionOpInterface entryPointFn, ArrayRef<Operation *> computeOps,
    TilePassPipeline passPipeline, LowerToAIEPassPipeline useLowerToAIEPipeline,
    AMDAIEDevice targetDevice, uint32_t numRows, uint32_t numCols) {
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

  if (failed(setRootConfigImpl(entryPointFn, rootOperation, passPipeline,
                               useLowerToAIEPipeline, targetDevice, numRows,
                               numCols)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult initAIELaunchConfig(FunctionOpInterface funcOp,
                                  TilePassPipeline passPipeline,
                                  LowerToAIEPassPipeline useLowerToAIEPipeline,
                                  AMDAIEDevice targetDevice, uint32_t numRows,
                                  uint32_t numCols) {
  if (getTranslationInfo(funcOp)) return success();

  // TODO (nmeshram): Need a default pipeline for control flow cases.
  if (funcOp.empty() || !llvm::hasSingleElement(funcOp.getFunctionBody()))
    return funcOp.emitError("Control flow not yet supported.");

  SmallVector<Operation *> computeOps = getComputeOps(funcOp);
  if (failed(setTranslationInfoAndRootConfig(funcOp, computeOps, passPipeline,
                                             useLowerToAIEPipeline,
                                             targetDevice, numRows, numCols)))
    return failure();

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(funcOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  return applyPatternsGreedily(funcOp, std::move(patterns));
}

}  // namespace mlir::iree_compiler::AMDAIE
