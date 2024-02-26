// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/KernelDispatch.h"

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "kernel-dispatch"

namespace mlir::iree_compiler::AMDAIE {

static LogicalResult setRootConfigForPadPipeline(func::FuncOp entryPointFn,
                                                 linalg::MatmulOp matmulOp) {
  SmallVector<int64_t> TileSizeLevel0 = {8, 8};
  SmallVector<int64_t> TileSizeLevel1 = {4, 4};
  SmallVector<int64_t> TileSizeLevel2 = {0, 0, 4};
  TileSizesListType tileSizes = {TileSizeLevel0, TileSizeLevel1,
                                 TileSizeLevel2};
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, matmulOp, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::None);
}

static SmallVector<int64_t> getPackedSize(Operation *op) {
  // TODO: consider emiting an error/warning if the default sizes are used as a
  // fallback.
  const SmallVector<int64_t> defaultSizes{0, 0, 0, 4, 4, 8};
  auto matmulOp = dyn_cast<linalg::MatmulOp>(op);
  if (!matmulOp) {
    return defaultSizes;
  }

  auto getElementType = [](Value v) {
    return v.getType().cast<ShapedType>().getElementType();
  };

  auto elTypeLhs = getElementType(op->getOperand(0));
  auto elTypeRhs = getElementType(op->getOperand(1));
  auto elTypeAcc = getElementType(op->getResult(0));

  auto maybeInstructionSize =
      getAIEMatmulInstructionSize(elTypeLhs, elTypeRhs, elTypeAcc);

  if (failed(maybeInstructionSize)) {
    return defaultSizes;
  }

  auto instructionSize = maybeInstructionSize.value();

  SmallVector<int64_t> packedSizes(6, 0);
  std::copy(instructionSize.begin(), instructionSize.end(),
            packedSizes.begin() + 3);

  return packedSizes;
}

static LogicalResult setRootConfigForSimplePackPipeline(
    func::FuncOp entryPointFn, linalg::MatmulOp matmulOp) {
  // ------------------------------------------------------
  // -------------- Set lowering config -------------------
  // ------------------------------------------------------
  // Assume working on a 2x2 AIE array and make sure the tile size is not larger
  // than the input size.
  auto initType = matmulOp.getDpsInitOperand(0)->get().getType();
  auto initShape = llvm::cast<ShapedType>(initType).getShape();
  auto tileM0 = std::min((int)initShape[0], 64);
  auto tileN0 = std::min((int)initShape[1], 64);
  auto tileM1 = std::max((int)tileM0 / 2, 1);
  auto tileN1 = std::max((int)tileN0 / 2, 1);
  auto lhsType = matmulOp.getDpsInputOperand(0)->get().getType();
  auto lhsShape = llvm::cast<ShapedType>(lhsType).getShape();
  auto tileK = std::min((int)lhsShape[1] / 8, 4);

  SmallVector<int64_t> TileSizeLevel0 = {tileM0, tileN0};
  SmallVector<int64_t> TileSizeLevel1 = {0, 0, 0, tileM1, tileN1};
  SmallVector<int64_t> TileSizeLevel2 = {0, 0, 0, 0, 0, tileK};
  TileSizesListType tileSizes = {TileSizeLevel0, TileSizeLevel1,
                                 TileSizeLevel2};
  if (failed(setOpConfigAndEntryPointFnTranslation(
          entryPointFn, matmulOp, tileSizes,
          IREE::Codegen::DispatchLoweringPassPipeline::None))) {
    return failure();
  }
  // ------------------------------------------------------
  // --------------- Set packing config -------------------
  // ------------------------------------------------------
  MLIRContext *context = entryPointFn.getContext();
  // Pack level => 1.
  // Set constraints for pack size [M, N] from first level of tile sizes
  // Currently set pack size k as the input size K to avoid failure.
  int64_t kSize = lhsShape[1];
  SmallVector<int64_t> packedSizes = {tileM0, tileN0, kSize};
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
  packedSizes = getPackedSize(matmulOp);
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
  setPackingConfig(matmulOp, config);
  return success();
}

static LogicalResult setRootConfigForPackPipeline(func::FuncOp entryPointFn,
                                                  linalg::MatmulOp matmulOp,
                                                  AIEConfig cfg) {
  // ------------------------------------------------------
  // -------------- Set lowering config -------------------
  // ------------------------------------------------------
  if (!(cfg.num_cores == 1 || cfg.num_cores == 2 || cfg.num_cores == 4))
    return matmulOp.emitOpError("unhandled number of cores");
  SmallVector<int64_t> TileSizeLevel0 = {16, 64 * cfg.num_cores};
  SmallVector<int64_t> TileSizeLevel1 = {0, 0, 64};
  SmallVector<int64_t> TileSizeLevel2 = {1, 1};
  TileSizesListType tileSizes = {TileSizeLevel0, TileSizeLevel1,
                                 TileSizeLevel2};
  if (failed(setOpConfigAndEntryPointFnTranslation(
          entryPointFn, matmulOp, tileSizes,
          IREE::Codegen::DispatchLoweringPassPipeline::None))) {
    return failure();
  }
  // ------------------------------------------------------
  // --------------- Set packing config -------------------
  // ------------------------------------------------------
  MLIRContext *context = entryPointFn.getContext();
  // Pack level => 1.
  SmallVector<int64_t> packedSizes = {16, 64, 64};
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
  packedSizes = getPackedSize(matmulOp);
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
  setPackingConfig(matmulOp, config);
  return success();
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::MatmulOp matmulOp,
                                   AIEPassPipeline usePassPipeline,
                                   AIEConfig cfg) {
  assert(!getLoweringConfig(matmulOp) && "expected lowering_config is not set");
  auto linalgOp = cast<linalg::LinalgOp>(matmulOp.getOperation());
  unsigned numLoops = linalgOp.getNumLoops();
  {
    SmallVector<unsigned> dims;
    linalgOp.getReductionDims(dims);
    if (dims.size() != 1 || dims[0] != numLoops - 1) {
      return matmulOp.emitOpError(
          "expected to have exactly one reduction dim, and it is the innermost "
          "dim");
    }
  }

  // TODO (nmeshram) : This needs to be moved in a separate more generalized
  // logic. Also, need a flag to experiment between pad based and pack based
  // approach which will have different tile sizes and pass pipelines
  if (usePassPipeline == AIEPassPipeline::PadPipeline)
    return setRootConfigForPadPipeline(entryPointFn, matmulOp);
  if (usePassPipeline == AIEPassPipeline::SimplePackPipeline)
    return setRootConfigForSimplePackPipeline(entryPointFn, matmulOp);
  if (usePassPipeline == AIEPassPipeline::PackPipeline)
    return setRootConfigForPackPipeline(entryPointFn, matmulOp, cfg);
  return matmulOp.emitOpError("unhandled pass pipeline");
}

/// Redirects to methods that set the configuration based on operation type.
static LogicalResult setRootConfigImpl(func::FuncOp entryPointFn, Operation *op,
                                       AIEPassPipeline usePassPipeline,
                                       AIEConfig cfg) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        // TODO (nmeshram): This is very limited for now, plan is to
        // let it first crash for all the other ops and then consiously
        // add support for them, this way we can verify our work.
        .Case<linalg::MatmulOp>([&](auto op) {
          return setRootConfig(entryPointFn, op, usePassPipeline, cfg);
        })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult setTranslationInfoAndRootConfig(
    func::FuncOp entryPointFn, ArrayRef<Operation *> computeOps,
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

LogicalResult initAIELaunchConfig(ModuleOp moduleOp,
                                  AIEPassPipeline usePassPipeline,
                                  AIEConfig cfg) {
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp) continue;
    if (getTranslationInfo(exportOp)) continue;

    // TODO (nmeshram): Need a default pipeline for control flow cases.
    if (funcOp.getBody().empty() || !llvm::hasSingleElement(funcOp.getBody())) {
      return funcOp.emitError("control flow not yet supported.");
    }

    SmallVector<Operation *> computeOps = getComputeOps(funcOp);
    if (failed(setTranslationInfoAndRootConfig(funcOp, computeOps,
                                               usePassPipeline, cfg))) {
      return failure();
    }
  }

  // The root configuration setting introduces `tensor.dim` operations.
  // Resolve those away.
  RewritePatternSet patterns(moduleOp.getContext());
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  return applyPatternsAndFoldGreedily(moduleOp, std::move(patterns));
}

}  // namespace mlir::iree_compiler::AMDAIE
