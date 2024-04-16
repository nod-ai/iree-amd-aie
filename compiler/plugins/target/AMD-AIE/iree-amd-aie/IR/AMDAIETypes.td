// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_DIALECT_AMDAIETYPES
#define IREE_COMPILER_AMDAIE_DIALECT_AMDAIETYPES

include "mlir/IR/AttrTypeBase.td"
include "iree-amd-aie/IR/AMDAIEDialect.td"

class AMDAIEDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AMDAIE_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def AMDAIE_LogicalObjectFifoType :
    AMDAIEDialect_Type<"LogicalObjectFifo", "logicalobjectfifo"> {
  let summary = "The logical objectfifo type encapsulating a memref";
  let description = [{
    The logical objectfifo type encapulates a memref and provides synchronized
    access operations to retrieve the underlying memref. This type is similar and 
    based on the MLIR-AIE ObjectFifo type. For now, this type only works with
    static memrefs.

    Example:
    ```mlir
    !amdaie.logicalobjectfifo<memref<8x16xi32>>
    ```
  }];

  let parameters = (ins "MemRefType":$element_type);
  let genVerifyDecl = 1;

  let assemblyFormat = "`<` $element_type `>`";
  
  let builders = [
    TypeBuilderWithInferredContext<(ins "MemRefType":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];  
}

def AnyAMDAIELogicalObjectFifoType : AnyTypeOf<[AMDAIE_LogicalObjectFifoType]>;

#endif // IREE_COMPILER_AMDAIE_DIALECT_AMDAIETYPES
