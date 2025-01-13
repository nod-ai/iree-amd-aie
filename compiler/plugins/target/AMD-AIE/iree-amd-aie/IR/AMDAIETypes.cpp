// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIETypes.h"

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "iree-amd-aie/IR/AMDAIETypes.cpp.inc"

namespace mlir::iree_compiler::AMDAIE {

void AMDAIEDialect::initializeAMDAIETypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "iree-amd-aie/IR/AMDAIETypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// LogicalObjectFifoType
//===----------------------------------------------------------------------===//

LogicalResult LogicalObjectFifoType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::MemRefType elementType, unsigned depth) {
  if (llvm::any_of(elementType.getShape(), [](auto dimSize) {
        return ShapedType::isDynamic(dimSize);
      })) {
    return emitError() << "should encapsulate static memref";
  }
  if (depth < 1 || depth > 4) return emitError() << "depth should be in [1, 4]";
  return success();
}

}  // namespace mlir::iree_compiler::AMDAIE
