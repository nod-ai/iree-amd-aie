// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#define DEBUG_TYPE "iree-amdaie-loads-store-alignment-reset"

namespace mlir::iree_compiler::AMDAIE {

using namespace mlir;

namespace {

/// A pass that removes the alignment attribute from llvm.load and llvm.store
/// operations. As example, this pass will replace
///
///  ```
///  %20 = llvm.load %19 {alignment = 1 : i64} : !llvm.ptr -> vector<32xi8>
///  ```
///
///  with
///
///  ```
///  %20 = llvm.load %19 : !llvm.ptr -> vector<32xi8>
///  ```
///
/// The motivation for this is that the alignment on the llvm.load operation,
/// which is assigned in the pass `convert-vector-to-llvm` is currently too
/// conservative and if left as is, results in poor (and sometimes invalid)
/// scalarized code in peano.
///
/// The pass `convert-vector-to-llvm` does not seem to be doing any analysis to
/// choose the highest possible alignment. It lowers
///
/// ```
/// %11 = vector.transfer_read %collapse_shape[%10], %c0_i8 {in_bounds = [true]}
/// : memref<1024xi8>, vector<32xi8>
/// ```
///
/// to
///
/// ```
/// %20 = llvm.load %19 {alignment = 1 : i64} : !llvm.ptr -> vector<32xi8>
/// ```
///
/// even when it can be inferred that %10 is a multiple ot 32. The pass
/// seems to always just use the alignment of the element type.
///
/// By resetting the alignments that the `convert-vector-to-llvm` pass assigns,
/// the lowering/translation to LLVMIR assigns new alignments. In other words it
/// seems that the translation does not modify existing alignments on llvm.load
/// or llvm.store, but if there is no alignment present it assigns one.
///
/// Whereas the `convert-vector-to-llvm` pass assigns an alignment based on
/// the element-type, the translation to LLVMIR assigns an alignment based on
/// the vector width. For example, for a vector of 32 i8 values, the alignment
/// assigned is 32.
///
/// I can imagine cases where the llvm.load actually loads with an alignment
/// less than the vector width, for example if you're loading overlapping
/// vectors (for a convolution say):
///
/// iteration 1: load bytes 0-8.
/// iteration 2: load bytes 4-12.
/// iteration 3: load bytes 8-16.
///
/// in this case using the width of the vector (8 bytes) as the alignment would
/// be incorrect, as the loads are only 4-byte aligned. Future work: check if
/// LLVMIR lowering correctly handles this case, if not implement an analysis.
///
/// See also https://jira.xilinx.com/projects/AIECC/issues/AIECC-589

class AMDAIELoadStoreAlignmentReset
    : public impl::AMDAIELoadStoreAlignmentResetBase<
          AMDAIELoadStoreAlignmentReset> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override {
    getOperation()->walk([](Operation *op) {
      if (auto loadOp = dyn_cast<LLVM::LoadOp>(op)) {
        loadOp.setAlignment(std::optional<uint64_t>());
      } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(op)) {
        storeOp.setAlignment(std::optional<uint64_t>());
      }
    });
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIELoadStoreAlignmentResetPass() {
  return std::make_unique<AMDAIELoadStoreAlignmentReset>();
}

}  // namespace mlir::iree_compiler::AMDAIE
