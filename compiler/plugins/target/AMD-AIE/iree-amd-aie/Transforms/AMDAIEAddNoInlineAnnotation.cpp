// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#define DEBUG_TYPE "iree-amdaie-add-no-inline-annotation"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEAddNoInlineAnnotationPass
    : public impl::AMDAIEAddNoInlineAnnotationBase<
          AMDAIEAddNoInlineAnnotationPass> {
 public:
  void getDependentDialects(DialectRegistry &reg) const override {
    reg.insert<LLVM::LLVMDialect>();
  }
  AMDAIEAddNoInlineAnnotationPass() = default;
  AMDAIEAddNoInlineAnnotationPass(const AMDAIEAddNoInlineAnnotationPass &){};
  void runOnOperation() override;
};

void AMDAIEAddNoInlineAnnotationPass::runOnOperation() {
  Operation *parentOp = getOperation();
  parentOp->walk([&](LLVM::LLVMFuncOp funcOp) { funcOp.setNoInline(true); });
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAddNoInlineAnnotationPass() {
  return std::make_unique<AMDAIEAddNoInlineAnnotationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
