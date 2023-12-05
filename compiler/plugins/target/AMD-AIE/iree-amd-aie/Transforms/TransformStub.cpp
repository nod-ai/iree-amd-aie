// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

using namespace mlir;

using iree_compiler::buildPad;
using iree_compiler::buildTileFuseDistToForallWithTileSizes;
using iree_compiler::blockX;
using iree_compiler::blockY;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using transform::MatchOp;
using transform::SplitHandleOp;

/// Matches `args` within `targetH` and unpacks a number of handles `N`.
/// Assumes there are exactly `N` matched ops (but could be relaxed).
/// Returns the tuple of handles.
template <int N, typename... MatchingArgs>
auto matchAndUnpack(ImplicitLocOpBuilder &b, Value targetH,
                    MatchingArgs... args) {
  Value matchedH = b.create<MatchOp>(targetH, args...);
  auto matchOp = b.create<SplitHandleOp>(matchedH,
                                         /*numHandles=*/N);
  assert(matchOp->getNumResults() == N && "Unexpected number of results");
  std::array<Value, N> a;
  for (int64_t i = 0; i < N; ++i)
    a[i] = matchOp->getResult(i);
  return std::tuple_cat(a);
}

std::tuple<Value, Value>
mlir::iree_compiler::AMDAIE::buildTransformTileAndFuseStub(
    ImplicitLocOpBuilder &b, Value variantH, Value rootH,
    ArrayRef<int64_t> tileSizes, ArrayRef<Attribute> threadDimMapping,
    bool foldIfBranch) {
  TileToForallAndFuseAndDistributeResult res =
      buildTileFuseDistToForallWithTileSizes(
          /*builder=*/b,
          /*variantH=*/variantH,
          /*rootH=*/rootH,
          /*opsToFuseH=*/{},
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(threadDimMapping));
//   if (foldIfBranch) {
//     Value ifOpH = b.create<transform::MatchOp>(res.forallH,
//                                                scf::IfOp::getOperationName());
//     b.create<transform::TakeAssumedBranchOp>(
//         ifOpH, /*takeElseBranch=*/b.getUnitAttr());
//   }
  return std::make_tuple(res.tiledOpH, res.forallH);
}

// TODO: Modify this to add any `symName`. Should even upstream this!
void mlir::iree_compiler::createTransformRegion(
    func::FuncOp entryPoint, StrategyBuilderFn buildStrategy) {
  MLIRContext *ctx = entryPoint.getContext();
  Location loc = entryPoint.getLoc();
  OpBuilder b(ctx);
  b.setInsertionPointAfter(entryPoint);
  auto topLevelTransformModule = b.create<ModuleOp>(loc);
  topLevelTransformModule->setAttr(
      transform::TransformDialect::kWithNamedSequenceAttrName, b.getUnitAttr());
  Region &topLevelTransformRegion = topLevelTransformModule.getBodyRegion();
  b.setInsertionPointToStart(&topLevelTransformRegion.front());
  auto anyOpType = transform::AnyOpType::get(b.getContext());
  auto sequence = b.create<transform::NamedSequenceOp>(
      loc,
      /*symName=*/
      std::string(
          transform::TransformDialect::kTransformEntryPointSymbolName.str()),
      /*rootType*/ anyOpType,
      /*resultTypes=*/TypeRange{},
      /*bodyBuilder=*/[&](OpBuilder &b, Location loc, Value variantH) {
        ImplicitLocOpBuilder ib(loc, b);
        buildStrategy(ib, variantH);
        b.create<transform::YieldOp>(loc);
      });
  (void)sequence;

  LDBG("transformation script:\n");
  LDBG("verification: " << sequence.verify().succeeded() << "\n");
}

void buildCleanUpStrategy(ImplicitLocOpBuilder &b, Value variantH) {
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  // This would add all the required canonicalizations.
  mlir::iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);
}

void cleanUpTransform() {
  iree_compiler::createTransformRegion(entryPoint, [&](ImplicitLocOpBuilder &b,
                                                       Value variantH) {
    return iree_compiler::AMDAIE::buildCleanUpStrategy(b, variantH);
  });
}

void buildMainTransformSequenceStrategy() {
    // LEVEL 1 TILING:
        // Step 1. Match linalg.fill and linalg.matmul.
        matchAndUnpack(variantOp);

        // Step 2. Tile linalg.matmul and fuse linalg.fill in the forall loop.
        MLIRContext *ctx = b.getContext();
        SmallVector<Attribute> blockDimMapping{blockX(ctx), blockY(ctx), blockZ(ctx)};
        blockDimMapping.resize(workgroupTileSizes.size());
        TileToForallAndFuseAndDistributeResult tileResult =
            buildTileFuseDistToForallWithTileSizes(
                /*builder=*/b,
                /*variantH=*/variantH,
                /*rootH=*/fusionTargetH,
                /*opsToFuseH=*/fusionGroupH,
                /*tileSizes=*/
                getAsOpFoldResult(b.getI64ArrayAttr(workgroupTileSizes)),
                /*threadDimMapping=*/b.getArrayAttr(blockDimMapping));

        // Step 3. Handle the workgroup count region.
        b.create<iree_compiler::IREE::transform_dialect::IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp>(
            tileResult.forallH);

        // Step 4. Pad the tiled linalg.matmul.
        // Use linalg::transform::PadOp since there you can add the copy_back_op data,
        // also you can get padded, pad and copy_back ops too.
        // TODO: Need to check how to invoke the same.
        b.create<transform::PadOp>(...);
        // DiagnosedSilenceableFailure
        // transform::PadOp::apply(transform::TransformRewriter &rewriter,
        //                         transform::TransformResults &results,
        //                         transform::TransformState &state)

        // Step 5. Rewrite the pad op obtained in step 4. in DPS.
        copyBackOpH =
            b.create<linalg::transform::RewriteInDestinationPassingStyleOp>(resH.getType(), resH);

        // Step 6. Promote operands and result to shared memory.
        // For each operand and result:
        //      do:
                    transform::BufferizeToAllocationOp::apply(
                    transform::TransformRewriter &rewriter,
                    transform::TransformResults &results, transform::TransformState &state)
    
    // Invoke the cleanup transform region using transform::IncludeOp.
    b.create<transform::IncludeOp>(...);

    // LEVEL 2 TILING:
        // Step 1. Match linalg.fill and linalg.matmul.
        matchAndUnpack(tileResult.forallH);

        // Step 2. Tile linalg.matmul and fuse linalg.fill in the forall loop.
        MLIRContext *ctx = b.getContext();
        SmallVector<Attribute> blockDimMapping{blockX(ctx), blockY(ctx), blockZ(ctx)};
        blockDimMapping.resize(workgroupTileSizes.size());
        TileToForallAndFuseAndDistributeResult tileResult =
            buildTileFuseDistToForallWithTileSizes(
                /*builder=*/b,
                /*variantH=*/variantH,
                /*rootH=*/fusionTargetH,
                /*opsToFuseH=*/fusionGroupH,
                /*tileSizes=*/
                getAsOpFoldResult(b.getI64ArrayAttr(workgroupTileSizes)),
                /*threadDimMapping=*/b.getArrayAttr(blockDimMapping));

        // Step 3. Pad the tiled linalg.matmul.
        // Use linalg::transform::PadOp since there you can add the copy_back_op data,
        // also you can get padded, pad and copy_back ops too.
        // TODO: Need to check how to invoke the same.
        b.create<transform::PadOp>(...);

        // Step 4. Rewrite the pad op obtained in step 3. in DPS.
        copyBackOpH =
            b.create<linalg::transform::RewriteInDestinationPassingStyleOp>(resH.getType(), resH);

        // Step 5. Promote the result to local memory.
        transform::BufferizeToAllocationOp::apply(
        transform::TransformRewriter &rewriter,
        transform::TransformResults &results, transform::TransformState &state)

    // Invoke the cleanup transform region using transform::IncludeOp.
    b.create<transform::IncludeOp>(...);

    // LEVEL 3 TILING:
        // Step 1.
        auto tiletoScfForOp = b.create<transform::TileUsingForOp>(rootH, tileSizes);

    // Invoke the cleanup transform region using transform::IncludeOp.
    b.create<transform::IncludeOp>(...);

        // Step 2. Pad the tiled linalg.matmul.
        // Use linalg::transform::PadOp since there you can add the copy_back_op data,
        // also you can get padded, pad and copy_back ops too.
        // TODO: Need to check how to invoke the same.
        b.create<transform::PadOp>(...);

        // Step 3. Rewrite the pad op obtained in step 2. in DPS.
        copyBackOpH =
            b.create<linalg::transform::RewriteInDestinationPassingStyleOp>(resH.getType(), resH);

        // Step 4. Promote operands to local memory.
        // For each operand:
        //      do:
                    transform::BufferizeToAllocationOp::apply(
                    transform::TransformRewriter &rewriter,
                    transform::TransformResults &results, transform::TransformState &state)

    // Invoke the cleanup transform region using transform::IncludeOp.
    b.create<transform::IncludeOp>(...);

    b.create<IREEEliminateEmptyTensorsOp>(variantH);
    variantH = b.create<IREEBufferizeOp>(variantH, targetGpu);

    // Invoke the cleanup transform region using transform::IncludeOp.
    b.create<transform::IncludeOp>(...);

}