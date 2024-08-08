// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_AIEX_DIALECT_H
#define MLIR_AIEX_DIALECT_H

#include "AIEDialect.h"

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

#include "aie/AIEXDialect.h.inc"

#define GET_OP_CLASSES
#include "aie/AIEX.h.inc"

#endif
