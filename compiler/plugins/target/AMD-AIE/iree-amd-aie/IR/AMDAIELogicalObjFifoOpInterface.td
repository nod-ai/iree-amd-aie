// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMDAIE_DIALECT_LOGICALOBJFIFOOPINTERFACE
#define IREE_AMDAIE_DIALECT_LOGICALOBJFIFOOPINTERFACE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CopyOpInterface.td"

//===----------------------------------------------------------------------===//
// Defines the interface for logical objectFifo operations.
//===----------------------------------------------------------------------===//

def LogicalObjFifoOpInterface : OpInterface<"LogicalObjFifoOpInterface"> {
  let description = [{
    Interface for operations creating a logical objectFifo.
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the buffer depth of the logical objectFifo. (E.g. 1 == "
               "single buffer, 2 == double buffer)",
      /*retTy=*/"uint8_t",
      /*methodName=*/"getDepth",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getDepth();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the assigned tiles.",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getTiles",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getTiles();
      }]
    >
  ];
}

#endif // IREE_AMDAIE_DIALECT_LOGICALOBJFIFOOPINTERFACE
