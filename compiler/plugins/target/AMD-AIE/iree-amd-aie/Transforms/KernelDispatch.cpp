// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/KernelDispatch.h"

#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "kernel-dispatch"
namespace mlir::iree_compiler::AMDAIE {

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(
    func::FuncOp entryPointFn, linalg::ContractionOpInterface contractionOp) {
  assert(!getLoweringConfig(contractionOp) &&
         "expected lowering_config is not set");
  auto linalgOp = cast<linalg::LinalgOp>(contractionOp.getOperation());
  unsigned numLoops = linalgOp.getNumLoops();
  {
    SmallVector<unsigned> dims;
    linalgOp.getReductionDims(dims);
    if (dims.size() != 1 || dims[0] != numLoops - 1) {
      return contractionOp.emitOpError(
          "expected to have exactly one reduction dim, and it is the innermost "
          "dim");
    }
  }

  // Make sure the tile size is not larger than the input size.
  auto initType = linalgOp.getDpsInitOperand(0)->get().getType();
  auto initShape = llvm::cast<ShapedType>(initType).getShape();
  auto tileM = std::min(initShape[1], (int64_t)64);
  auto tileN = std::min(initShape[2], (int64_t)64);

  SmallVector<int64_t> TileSizeLevel0 = {1, tileM, tileN};
  SmallVector<int64_t> TileSizeLevel1 = {1, 1, 1};
  SmallVector<int64_t> TileSizeLevel2 = {0, 0, 0, 1};
  TileSizesListType tileSizes = {TileSizeLevel0, TileSizeLevel1,
                                 TileSizeLevel2};
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, linalgOp, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::None);

  // #config = #iree_codegen.lowering_config<tile_sizes = [[1, 64, 64, 0], [1,
  // 16, 4, 0], [0, 0, 0, 64]]>
}

/// Sets the lowering configuration for dispatch region with root op that
/// implements the contraction operation interface.
static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   linalg::MatmulOp matmulOp,
                                   AIEPassPipeline usePassPipeline) {
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
  if (usePassPipeline == AIEPassPipeline::PadPipeline) {
    SmallVector<int64_t> TileSizeLevel0 = {8, 8};
    SmallVector<int64_t> TileSizeLevel1 = {4, 4};
    SmallVector<int64_t> TileSizeLevel2 = {0, 0, 4};
    TileSizesListType tileSizes = {TileSizeLevel0, TileSizeLevel1,
                                   TileSizeLevel2};
    return setOpConfigAndEntryPointFnTranslation(
        entryPointFn, matmulOp, tileSizes,
        IREE::Codegen::DispatchLoweringPassPipeline::None);
  } else if (usePassPipeline == AIEPassPipeline::SimplePackPipeline) {
    // Make sure the tile size is not larger than the input size.
    auto initType = matmulOp.getDpsInitOperand(0)->get().getType();
    auto initShape = llvm::cast<ShapedType>(initType).getShape();
    auto tileM = std::min(initShape[0], (int64_t)64);
    auto tileN = std::min(initShape[1], (int64_t)64);

    SmallVector<int64_t> TileSizeLevel0 = {tileM, tileN};
    SmallVector<int64_t> TileSizeLevel1 = {1, 1};
    SmallVector<int64_t> TileSizeLevel2 = {0, 0, 1};
    TileSizesListType tileSizes = {TileSizeLevel0, TileSizeLevel1,
                                   TileSizeLevel2};
    return setOpConfigAndEntryPointFnTranslation(
        entryPointFn, matmulOp, tileSizes,
        IREE::Codegen::DispatchLoweringPassPipeline::None);
  } else if (usePassPipeline == AIEPassPipeline::PackPipeline) {
    SmallVector<int64_t> TileSizeLevel0 = {16, 64};
    SmallVector<int64_t> TileSizeLevel1 = {0, 0, 64};
    SmallVector<int64_t> TileSizeLevel2 = {1, 1};
    TileSizesListType tileSizes = {TileSizeLevel0, TileSizeLevel1,
                                   TileSizeLevel2};
    return setOpConfigAndEntryPointFnTranslation(
        entryPointFn, matmulOp, tileSizes,
        IREE::Codegen::DispatchLoweringPassPipeline::None);
  }
  return matmulOp.emitOpError("unhandled pass pipeline");
}

/// Redirects to methods that set the configuration based on operation type.
static LogicalResult setRootConfigImpl(func::FuncOp entryPointFn, Operation *op,
                                       AIEPassPipeline usePassPipeline) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        // TODO (nmeshram): This is very limited for now, plan is to
        // let it first crash for all the other ops and then consiously
        // add support for them, this way we can verify our work.
        .Case<linalg::ContractionOpInterface>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Case<linalg::MatmulOp>([&](auto op) {
          return setRootConfig(entryPointFn, op, usePassPipeline);
        })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult setTranslationInfoAndRootConfig(
    func::FuncOp entryPointFn, ArrayRef<Operation *> computeOps,
    AIEPassPipeline usePassPipeline) {
  // Make sure that lowering_config is not preset on any compute ops.
  for (auto computeOp : computeOps) {
    if (getLoweringConfig(computeOp)) return failure();
  }

  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp)) return failure();
  Operation *rootOperation = rootOp.value();
  // llvm::outs()<<"ROOT OPERATION : "<<*rootOperation<<"\n";
  // assert(false);

  // TODO (nmeshram): Handle the case with no known root operation.
  if (!rootOperation) {
    return entryPointFn.emitError("Case with no root ops not yet supported.");
  }

  if (failed(setRootConfigImpl(entryPointFn, rootOperation, usePassPipeline))) {
    return failure();
  }

  // TODO (nmeshram): // Set vector level tile sizes for other operations
  // individually.

  return success();
}

LogicalResult initAIELaunchConfig(ModuleOp moduleOp,
                                  AIEPassPipeline usePassPipeline) {
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
                                               usePassPipeline))) {
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
