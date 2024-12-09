// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file composes more complex strided DMA ops by iteratively:
// 1. Combining ops in the same block.
// 2. Subsuming loop iterations into the strided access pattern.
//
//===----------------------------------------------------------------------===//

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Transforms.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEDmaUtils.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-dma-composition"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEDmaCompositionPass
    : public impl::AMDAIEDmaCompositionBase<AMDAIEDmaCompositionPass> {
 public:
  AMDAIEDmaCompositionPass() = default;
  AMDAIEDmaCompositionPass(const AMDAIEDmaCompositionPass &pass){};
  AMDAIEDmaCompositionPass(const AMDAIEDmaCompositionOptions &options)
      : AMDAIEDmaCompositionBase(options) {}
  void runOnOperation() override;
};

void AMDAIEDmaCompositionPass::runOnOperation() {
  Operation *parentOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration. This "
           "device-specific information is required to determine when loops "
           "can be subsumed into DMA operations, and must be attached to a "
           "containing ModuleOp.";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());
  populateDmaLoopSubsumptionPattern(patterns, deviceModel,
                                    onlyZeroStrideOnOuterDim);
  populateStridedOpCombinationPattern(patterns);
  populateCanonicalizeDoublyStridedOpPatterns(patterns, false, deviceModel);

  if (failed(applyPatternsAndFoldGreedily(parentOp, std::move(patterns)))) {
    parentOp->emitOpError("failed to compose strided operations");
    return signalPassFailure();
  }

  IRRewriter rewriter(parentOp->getContext());
  if (failed(moveNpuDmaSyncUsersAfterAncestorInSameBlock(rewriter, parentOp))) {
    parentOp->emitOpError() << "failed to move DMA users to correct scope "
                               "after strided op composition";
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEDmaCompositionPass(
    AMDAIEDmaCompositionOptions options) {
  return std::make_unique<AMDAIEDmaCompositionPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
