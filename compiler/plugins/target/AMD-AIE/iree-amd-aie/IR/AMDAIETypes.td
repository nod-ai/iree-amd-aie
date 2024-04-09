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
    DialectType<AMDAIE_Dialect, CPred<"$_self.isa<AMDAIELogicalObjectFifoType>()">,
                "AMDAIE objectFifo type">;

def AnyAMDAIELogicalObjectFifoType : AnyTypeOf<[AMDAIE_LogicalObjectFifoType]>;

#endif // IREE_COMPILER_AMDAIE_DIALECT_AMDAIETYPES
