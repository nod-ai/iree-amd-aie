//===- AIEDialect.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIE_DIALECT_H
#define MLIR_AIE_DIALECT_H

#include "AIEEnums.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

namespace xilinx::AIE {
template <typename T>
bool hasName(T &op) {
  return bool(op.getOperation()->template getAttrOfType<mlir::StringAttr>(
      mlir::SymbolTable::getSymbolAttrName()));
}

template <typename T>
mlir::StringAttr name(T &op) {
  if (auto attr = op.getOperation()->template getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName()))
    return attr;
  op.emitOpError("does not have '")
      << mlir::SymbolTable::getSymbolAttrName() << "' attribute specified";
  llvm::report_fatal_error("couldn't get name");
}

class TileOp;
}  // namespace xilinx::AIE

/// Include the generated interface declarations.
#include "aie/AIEInterfaces.h.inc"

namespace xilinx::AIE {
mlir::LogicalResult myVerifyOffsetSizeAndStrideOp(
    mlir::OffsetSizeAndStrideOpInterface op);
template <typename ConcreteOp>
struct MyOffsetSizeAndStrideOpInterfaceTrait
    : public ::mlir::detail::OffsetSizeAndStrideOpInterfaceTrait<ConcreteOp> {
  static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
    return myVerifyOffsetSizeAndStrideOp(
        ::mlir::cast<::mlir::OffsetSizeAndStrideOpInterface>(op));
  }
};

struct MyOffsetSizeAndStrideOpInterface
    : ::mlir::OffsetSizeAndStrideOpInterface {
  template <typename ConcreteOp>
  struct Trait : public MyOffsetSizeAndStrideOpInterfaceTrait<ConcreteOp> {};
};
}  // namespace xilinx::AIE

// Include dialect declarations such as parseAttributes, parseType
#include "aie/AIEDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aie/AIEAttrs.h.inc"

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "aie/AIEOps.h.inc"

namespace xilinx::AIE {
template <typename T>
inline TileOp getTileOp(T op) {
  return llvm::cast<TileOp>(op.getTile().getDefiningOp());
}

}  // namespace xilinx::AIE

#endif
