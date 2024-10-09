// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#define DEBUG_TYPE "iree-amdaie-acquire-release-to-use-lock"

namespace mlir::iree_compiler::AMDAIE {

  using namespace mlir;

namespace {

// A pass which removes the alignment attribute from llvm load operations, 
// if the alignment is less than 4 (2 or 1).
//
// Example. The pass replaces:
//
// ```
//  %113 = llvm.load %112 {alignment = 2 : i64} 
//                   : !llvm.ptr -> vector<32xbf16>
// ```
//
// with
//
// ```
//  %113 = llvm.load %112 
//                   : !llvm.ptr -> vector<32xbf16>
// ```
//
// If this pass is not included in the matmul pipeline, there is an OOM error
// later in the compilation. This is a temporary workaround while a better
// solution is found: propagation of memref.assume_alignment is one option. 
// See also https://jira.xilinx.com/projects/AIECC/issues/AIECC-589

class AMDAIELoadAlignmentReset
    : public impl::AMDAIELoadAlignmentResetBase<
          AMDAIELoadAlignmentReset> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override {
    getOperation()->walk([](Operation *op) {
      if (auto loadOp = dyn_cast<LLVM::LoadOp>(op)) {
        auto alignmentAttr = loadOp.getAlignmentAttr();
        if (alignmentAttr) {
          int alignmentVal = alignmentAttr.getValue().getSExtValue();
          if (alignmentVal == 2 || alignmentVal == 1) {
            loadOp.setAlignment(std::optional<uint64_t>());
          }
        }
      }
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIELoadAlignmentResetPass() {
  return std::make_unique<AMDAIELoadAlignmentReset>();
}

}  // namespace mlir::iree_compiler::AMDAIE
