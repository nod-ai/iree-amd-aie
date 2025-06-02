// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#define DEBUG_TYPE "iree-amdaie-remove-wrap-flag-from-gep"

namespace mlir::iree_compiler::AMDAIE {

using namespace mlir;

namespace {

/// Remove the wrap-flag attribute from llvm's getelementptr
///
/// - LLVM::GEPNoWrapFlags::nusw
/// - LLVM::GEPNoWrapFlags::nuw
/// - LLVM::GEPNoWrapFlags::inbounds
///
/// These were introduced to MLIR in
///
/// https://github.com/llvm/llvm-project/pull/137272
///
/// and introduced to LLVM about July 2024
///
/// https://discourse.llvm.org/t/rfc-add-nusw-and-nuw-flags-for-getelementptr/78672
///
/// Peano is a fork of LLVM preceding the addition of these attributes.
///
/// So peano cannot ingest LLVM IR with these attributes.
///
/// So to be able to target peano from here, we remove this attribute.
///
/// This workaround should be removed when peano gets rebased onto a more
/// recent point in the history of llvm-project.
class AMDAIERemoveWrapFlagFromGep
    : public impl::AMDAIERemoveWrapFlagFromGepBase<
          AMDAIERemoveWrapFlagFromGep> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    getOperation()->walk([](Operation *op) {
      if (auto gepOp = dyn_cast<LLVM::GEPOp>(op)) {
        gepOp.setNoWrapFlags(LLVM::GEPNoWrapFlags::none);
      }
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIERemoveWrapFlagFromGepPass() {
  return std::make_unique<AMDAIERemoveWrapFlagFromGep>();
}

}  // namespace mlir::iree_compiler::AMDAIE
