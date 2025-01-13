// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_DIALECT_AMDAIETYPES
#define IREE_COMPILER_AMDAIE_DIALECT_AMDAIETYPES

include "iree-amd-aie/IR/AMDAIEDialect.td"
include "mlir/IR/AttrTypeBase.td"

class AMDAIEDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AMDAIE_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// The types for async tokens which can be returned from async operations. The
// dedicated types for `source` and `target` can be used to specify on which
// side of (for example a copy/DMA) operation should be synchronized if there
// are multiple.
def AMDAIE_AsyncTokenType : AMDAIEDialect_Type<"AsyncToken", "async_token">;
def AMDAIE_AsyncSourceTokenType : AMDAIEDialect_Type<"AsyncSourceToken", "async_source_token">;
def AMDAIE_AsyncTargetTokenType : AMDAIEDialect_Type<"AsyncTargetToken", "async_target_token">;
def AMDAIE_AnyAsyncTokenType
  : AnyTypeOf<[AMDAIE_AsyncTokenType, AMDAIE_AsyncSourceTokenType, AMDAIE_AsyncTargetTokenType]>;

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

  let parameters = (ins "MemRefType":$element_type,
                        DefaultValuedParameter<"unsigned", "1">:$depth);
  let genVerifyDecl = 1;

  let assemblyFormat = "`<` $element_type (`,` $depth^)? `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "MemRefType":$elementType), [{
      return $_get(elementType.getContext(), elementType, 1);
    }]>,
    TypeBuilderWithInferredContext<(ins "MemRefType":$elementType, "unsigned":$depth), [{
      return $_get(elementType.getContext(), elementType, depth);
    }]>
  ];

  let extraClassDeclaration = [{
    Attribute getMemorySpace() {
      return getElementType().getMemorySpace();
    }

    /// Helper method to return the memory space as an integer. If no memory
    /// space attribute exists, this indicates a global memory space and we
    /// return 0. Else we cast the memory space attribute to an integer.
    uint8_t getMemorySpaceAsUInt() {
      Attribute memSpace = getMemorySpace();
      return memSpace ? llvm::cast<IntegerAttr>(memSpace).getInt() : 0;
    }
  }];
}

def AnyAMDAIELogicalObjectFifoType : AnyTypeOf<[AMDAIE_LogicalObjectFifoType]>;

#endif // IREE_COMPILER_AMDAIE_DIALECT_AMDAIETYPES
