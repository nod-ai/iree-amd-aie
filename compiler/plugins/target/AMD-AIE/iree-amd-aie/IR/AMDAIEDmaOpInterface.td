// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMDAIE_DIALECT_DMAOPINTERFACE
#define IREE_AMDAIE_DIALECT_DMAOPINTERFACE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Defines the interface for dma-like operations.
//===----------------------------------------------------------------------===//

def DoublyStridedOpInterface : OpInterface<"DoublyStridedInterface"> {
  let description = [{
    An operation with strided access on both source and target.
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getTargetOffsets"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getTargetSizes"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getTargetStrides"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getSourceOffsets"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getSourceSizes"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getSourceStrides"
    >
  ];
}

#endif // IREE_AMDAIE_DIALECT_DMAOPINTERFACE
