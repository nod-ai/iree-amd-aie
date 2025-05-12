// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#define DEBUG_TYPE "iree-amdaie-insert-copies"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEInsertCopiesPass
    : public impl::AMDAIEInsertCopiesBase<AMDAIEInsertCopiesPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AMDAIEDialect>();
  }

  AMDAIEInsertCopiesPass() = default;
  AMDAIEInsertCopiesPass(const AMDAIEInsertCopiesPass &pass){};
  void runOnOperation() override;
};

void AMDAIEInsertCopiesPass::runOnOperation() {}

}  // namespace

std::unique_ptr<Pass> createAMDAIEInsertCopiesPass() {
  return std::make_unique<AMDAIEInsertCopiesPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
