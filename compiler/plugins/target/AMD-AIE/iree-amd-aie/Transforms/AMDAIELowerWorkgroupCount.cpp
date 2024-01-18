// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {
class AMDAIELowerWorkgroupCountPass
    : public AMDAIELowerWorkgroupCountBase<AMDAIELowerWorkgroupCountPass> {
  void runOnOperation() override;
};
}  // namespace

/// Lower workgroup count operations to all 1s.
static LogicalResult lowerWorkgroupCount(
    RewriterBase &rewriter, IREE::HAL::ExecutableExportOp exportOp) {
  Block *body = exportOp.getWorkgroupCountBody();
  if (!body) {
    return exportOp.emitOpError("unexpected empty workgroup count region");
  }
  OpBuilder::InsertionGuard g(rewriter);
  for (auto workgroupCountOp : llvm::make_early_inc_range(
           body->getOps<IREE::Flow::DispatchWorkgroupCountFromSliceOp>())) {
    rewriter.setInsertionPoint(workgroupCountOp);
    Location loc = workgroupCountOp.getLoc();
    SmallVector<OpFoldResult> results;
    results.resize(workgroupCountOp.getNumResults(), rewriter.getIndexAttr(1));
    rewriter.replaceOp(workgroupCountOp,
                       getValueOrCreateConstantIndexOp(rewriter, loc, results));
  }
  return success();
}

void AMDAIELowerWorkgroupCountPass::runOnOperation() {
  IREE::HAL::ExecutableVariantOp variantOp = getOperation();
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  for (auto entryPointOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
    if (failed(lowerWorkgroupCount(rewriter, entryPointOp))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createAMDAIELowerWorkgroupCountPass() {
  return std::make_unique<AMDAIELowerWorkgroupCountPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
