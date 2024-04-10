// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_OP_CLASSES
#include "iree-amd-aie/IR/AMDAIEOps.cpp.inc"

namespace mlir::iree_compiler::AMDAIE {

void AMDAIEDialect::initializeAMDAIEOps() {
  addOperations<
#define GET_OP_LIST
#include "iree-amd-aie/IR/AMDAIEOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DmaCpyNdOp
//===----------------------------------------------------------------------===//

LogicalResult DmaCpyNdOp::verify() {
  if (failed(verifyCommon()))
    return failure();
  if (!isa<LogicalObjectFifoFromMemrefOp>(getSource().getDefiningOp()))
    return emitOpError("should have a `LogicalObjectFifoFromMemrefOp` as source");
  if (!isa<LogicalObjectFifoFromMemrefOp>(getTarget().getDefiningOp()))
    return emitOpError("should have a `LogicalObjectFifoFromMemrefOp` as target");
  return success();
}

LogicalObjectFifoFromMemrefOp DmaCpyNdOp::getSourceObjectFifo() {
  return dyn_cast<LogicalObjectFifoFromMemrefOp>(getSource().getDefiningOp());
};

LogicalObjectFifoFromMemrefOp DmaCpyNdOp::getTargetObjectFifo() {
  return dyn_cast<LogicalObjectFifoFromMemrefOp>(getTarget().getDefiningOp());
};

}  // namespace mlir::iree_compiler::AMDAIE
