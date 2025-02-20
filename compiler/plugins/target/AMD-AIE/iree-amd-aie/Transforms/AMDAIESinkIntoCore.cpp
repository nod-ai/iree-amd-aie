// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "iree-amdaie-sink-into-core"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIESinkIntoCorePass
    : public impl::AMDAIESinkIntoCoreBase<AMDAIESinkIntoCorePass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect,
                    xilinx::AIE::AIEDialect, AMDAIE::AMDAIEDialect>();
  }
  void runOnOperation() override {
    auto shouldSink = [&](Operation *op) -> bool {
      // Ops in the amdaie dialect are probably related to data movement
      // and should not be sunk into the core. This might need adjustment
      // later.
      if (op->getDialect()->getNamespace() ==
          AMDAIE::AMDAIEDialect::getDialectNamespace()) {
        return false;
      }
      return true;
    };
    IRRewriter rewriter(getOperation());
    SmallVector<AMDAIE::CoreOp> coreOps;

    getOperation()->walk(
        [&](AMDAIE::CoreOp coreOp) { coreOps.push_back(coreOp); });
    for (auto coreOp : coreOps) {
      sinkInto(coreOp.getRegion(), rewriter, shouldSink);
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIESinkIntoCorePass() {
  return std::make_unique<AMDAIESinkIntoCorePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
