// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#define DEBUG_TYPE "iree-amdaie-add-no-alias-function-arguments"

namespace mlir::iree_compiler::AMDAIE {

namespace {

SmallVector<std::pair<func::FuncOp, SmallVector<func::CallOp>>>
getFunctionsAndTheirCallers(Operation *rootOp) {
  // A mapping from all the function ops in the root op, to their callers.
  SmallVector<std::pair<func::FuncOp, SmallVector<func::CallOp>>>
      functionsAndCallers;

  // A mapping from function symbol names, to their index in
  // `functionsAndCallers`.
  DenseMap<StringRef, uint32_t> funcOpIndex;

  // Find all the function ops
  rootOp->walk([&](func::FuncOp funcOp) {
    funcOpIndex.insert({funcOp.getSymName(), functionsAndCallers.size()});
    SmallVector<func::CallOp> callers;
    functionsAndCallers.push_back({funcOp, callers});
  });

  // Add the callers to the mapping `functionsAndCallers`
  rootOp->walk([&](func::CallOp callOp) {
    StringRef callee = callOp.getCallee();
    auto iter = funcOpIndex.find(callee);
    if (iter != funcOpIndex.end()) {
      functionsAndCallers[iter->second].second.push_back(callOp);
    }
  });
  return functionsAndCallers;
}

/// Traverse backwards through the definition chain of first operands
/// starting from `initial`, until either a memref.alloc or a amdaie.buffer
/// operation is found. If neither is found, return a failure.
FailureOr<Operation *> getDefiningAllocation(Operation *initial) {
  Operation *current = initial;
  while (current) {
    if (isa<AMDAIE::BufferOp>(current) || isa<memref::AllocOp>(current)) {
      return current;
    }
    if (current->getNumOperands() != 1) {
      InFlightDiagnostic message =
          initial->emitOpError()
          << "could not be traced back to an allocation operation, "
          << "an operation with " << current->getNumOperands()
          << " operands was encountered while traversing defining ops.";
      return message;
    }
    current = current->getOperand(0).getDefiningOp();
  }
  return initial->emitOpError()
         << "could not be traced back to an allocation operation.";
};

/// Return a vector containing for every operand of `callOp`, a bool that is
/// true if the operand is an alias of a memref that does not alias with any
/// other operand.
FailureOr<SmallVector<bool>> getNonAliasingMemrefArguments(
    func::CallOp callOp) {
  // Find the allocations that define the memref operands of the call op.
  // This vector contains, for each memref operand, a pair containing
  // the allocation that defines it, and the index of the operand.
  SmallVector<std::pair<Operation *, uint32_t>> memrefAndIndex;
  for (auto [index, operand] : llvm::enumerate(callOp.getOperands())) {
    if (!isa<MemRefType>(operand.getType())) continue;
    if (operand.getDefiningOp() == nullptr) {
      return callOp->emitOpError(
          "has an operand with no defining op, failed to find allocation");
    }
    FailureOr<Operation *> maybeAllocation =
        getDefiningAllocation(operand.getDefiningOp());
    if (failed(maybeAllocation)) return failure();
    Operation *allocation = maybeAllocation.value();
    memrefAndIndex.push_back({allocation, index});
  }

  SmallVector<bool> nonAliasingMemref(callOp.getNumOperands(), false);
  for (auto [memref, index] : memrefAndIndex) {
    bool isAliasing = false;
    for (auto [otherMemref, otherIndex] : memrefAndIndex) {
      if (memref == otherMemref && index != otherIndex) {
        isAliasing = true;
      }
    }
    uint32_t operandIndex = index;
    nonAliasingMemref[operandIndex] = !isAliasing;
  }
  return nonAliasingMemref;
}

class AMDAIEAddNoAliasFunctionArgumentsPass
    : public impl::AMDAIEAddNoAliasFunctionArgumentsBase<
          AMDAIEAddNoAliasFunctionArgumentsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }
  AMDAIEAddNoAliasFunctionArgumentsPass() = default;
  AMDAIEAddNoAliasFunctionArgumentsPass(
      const AMDAIEAddNoAliasFunctionArgumentsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEAddNoAliasFunctionArgumentsPass::runOnOperation() {
  Operation *op = getOperation();

  // TODO(newling): resolve numerical issues on strix.
  {
    std::optional<AMDAIEDevice> device = getConfigAMDAIEDeviceFromAncestor(op);
    if (device.has_value() && isAie2P(device.value())) return;
  }

  IRRewriter rewriter(op);
  SmallVector<std::pair<func::FuncOp, SmallVector<func::CallOp>>>
      functionsAndCallers = getFunctionsAndTheirCallers(op);
  for (auto [func, callers] : functionsAndCallers) {
    uint32_t numOperands = func.getNumArguments();
    SmallVector<bool> nonAliasingMemref(numOperands, true);
    for (func::CallOp caller : callers) {
      assert(numOperands == caller.getNumOperands() &&
             "Number of operands in caller and callee do not match");
      FailureOr<SmallVector<bool>> maybeNonAliasingArguments =
          getNonAliasingMemrefArguments(caller);
      if (failed(maybeNonAliasingArguments)) {
        return signalPassFailure();
      }
      SmallVector<bool> nonAliasings = maybeNonAliasingArguments.value();
      for (uint32_t i = 0; i < nonAliasingMemref.size(); ++i) {
        nonAliasingMemref[i] = nonAliasingMemref[i] && nonAliasings[i];
      }
    }

    StringRef noAliasAttrName = LLVM::LLVMDialect::getNoAliasAttrName();
    ArrayRef<BlockArgument> args = func.getArguments();
    for (auto [index, _] : llvm::enumerate(args)) {
      if (nonAliasingMemref[index]) {
        func.setArgAttr(index, noAliasAttrName, rewriter.getUnitAttr());
      }
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEAddNoAliasFunctionArgumentsPass() {
  return std::make_unique<AMDAIEAddNoAliasFunctionArgumentsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
